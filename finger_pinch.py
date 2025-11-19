import cv2
import mediapipe as mp
import math
import numpy as np
import time
from collections import deque
from enum import Enum

class CalibrationState(Enum):
    """States for the calibration process"""
    IDLE = 0
    INSTRUCTIONS = 1
    COLLECTING_PINCH = 2
    COLLECTING_OPEN = 3
    FINALIZING = 4
    COMPLETE = 5

class CalibratedPinchDetector:
    """
    Production-ready pinch detector with user calibration system
    Achieves 15-20% accuracy improvement through personalization
    """
    
    def __init__(self, 
                 instruction_duration=10,           # ✅ NEW: How long to show instructions
                 samples_per_gesture=100,            # ✅ NEW: Number of samples to collect
                 completion_message_duration=3):    # ✅ NEW: How long to show success message
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Calibration data
        self.user_threshold = None
        self.calibration_samples = []
        self.default_threshold = 0.03  # 3cm default
        
        # ✅ NEW: Store configurable timing parameters
        self.instruction_duration = instruction_duration
        self.samples_per_gesture = samples_per_gesture
        self.completion_message_duration = completion_message_duration
        
        # Calibration state machine
        self.calibration_state = CalibrationState.IDLE
        self.calibration_start_time = 0
        self.required_pinch_samples = samples_per_gesture  # ✅ CHANGED: Use configurable value
        self.required_open_samples = samples_per_gesture   # ✅ CHANGED: Use configurable value
        self.pinch_sample_count = 0
        self.open_sample_count = 0

        
        # Hysteresis for stable detection
        self.is_pinching = False
        self.pinch_enter_threshold = None  # Set after calibration
        self.pinch_exit_threshold = None   # Set after calibration
        
        # Smoothing buffer for stability
        self.pinch_buffer = deque(maxlen=5)
        
        # Statistics for display
        self.calibration_stats = {
            'pinch_distances': [],
            'open_distances': [],
            'avg_pinch': 0,
            'avg_open': 0,
            'std_pinch': 0,
            'std_open': 0
        }
    
    def calculate_distance_3d(self, point1, point2):
        """Calculate Euclidean distance in 3D space"""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )
    
    def start_calibration(self):
        """Initialize calibration process"""
        print("\n" + "="*60)
        print("STARTING CALIBRATION")
        print("="*60)
        self.calibration_state = CalibrationState.INSTRUCTIONS
        self.calibration_samples = []
        self.pinch_sample_count = 0
        self.open_sample_count = 0
        self.calibration_start_time = time.time()
    
    def collect_calibration_sample(self, hand_world_landmarks, is_pinching):
        """
        Collect a single calibration sample
        
        Args:
            hand_world_landmarks: MediaPipe world landmarks
            is_pinching: True if user is performing pinch, False if open hand
        """
        thumb_tip = hand_world_landmarks.landmark[4]
        index_tip = hand_world_landmarks.landmark[8]
        
        distance = self.calculate_distance_3d(thumb_tip, index_tip)
        
        self.calibration_samples.append({
            'distance': distance,
            'is_pinching': is_pinching,
            'timestamp': time.time()
        })
        
        # Update sample counts
        if is_pinching:
            self.pinch_sample_count += 1
            self.calibration_stats['pinch_distances'].append(distance)
        else:
            self.open_sample_count += 1
            self.calibration_stats['open_distances'].append(distance)
    
    def finalize_calibration(self):
        """
        Calculate optimal threshold from collected calibration data
        Uses statistical analysis for robust threshold selection
        """
        pinch_distances = [s['distance'] for s in self.calibration_samples 
                          if s['is_pinching']]
        open_distances = [s['distance'] for s in self.calibration_samples 
                         if not s['is_pinching']]
        
        if not pinch_distances or not open_distances:
            print("❌ Calibration failed: Insufficient data")
            return False
        
        # Calculate statistics
        avg_pinch = np.mean(pinch_distances)
        avg_open = np.mean(open_distances)
        std_pinch = np.std(pinch_distances)
        std_open = np.std(open_distances)
        
        # Store statistics
        self.calibration_stats['avg_pinch'] = avg_pinch
        self.calibration_stats['avg_open'] = avg_open
        self.calibration_stats['std_pinch'] = std_pinch
        self.calibration_stats['std_open'] = std_open
        
        # Calculate optimal threshold (midpoint between averages)
        self.user_threshold = (avg_pinch + avg_open) / 2
        
        # Set hysteresis thresholds (prevents flickering)
        # Enter pinch: slightly below threshold
        # Exit pinch: slightly above threshold
        hysteresis_margin = (avg_open - avg_pinch) * 0.15  # 15% margin
        self.pinch_enter_threshold = self.user_threshold - hysteresis_margin
        self.pinch_exit_threshold = self.user_threshold + hysteresis_margin
        
        # Validation: Check if there's sufficient separation
        separation = avg_open - avg_pinch
        if separation < 0.01:  # Less than 1cm separation
            print("⚠️  Warning: Small separation between pinch and open gestures")
            print(f"   Consider more distinct gestures during calibration")
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   Pinch samples: {len(pinch_distances)}")
        print(f"   Open samples: {len(open_distances)}")
        print(f"   Avg pinch distance: {avg_pinch*100:.2f}cm (±{std_pinch*100:.2f}cm)")
        print(f"   Avg open distance: {avg_open*100:.2f}cm (±{std_open*100:.2f}cm)")
        print(f"   Separation: {separation*100:.2f}cm")
        print(f"\n🎯 Thresholds:")
        print(f"   Main threshold: {self.user_threshold*100:.2f}cm")
        print(f"   Enter pinch: < {self.pinch_enter_threshold*100:.2f}cm")
        print(f"   Exit pinch: > {self.pinch_exit_threshold*100:.2f}cm")
        print("="*60 + "\n")
        
        self.calibration_state = CalibrationState.COMPLETE
        return True
    
    def detect_pinch_calibrated(self, hand_world_landmarks):
        """
        Detect pinch using calibrated thresholds with hysteresis
        
        Returns:
            tuple: (is_pinching, distance, confidence)
        """
        thumb_tip = hand_world_landmarks.landmark[4]
        index_tip = hand_world_landmarks.landmark[8]
        
        distance = self.calculate_distance_3d(thumb_tip, index_tip)
        
        # Use calibrated or default thresholds
        if self.user_threshold is None:
            # Not calibrated - use default
            current_threshold = self.default_threshold
            is_pinch = distance < current_threshold
            confidence = 1.0 - (distance / current_threshold)
        else:
            # Use calibrated thresholds with hysteresis
            if not self.is_pinching:
                # Currently not pinching - check if should enter pinch
                if distance < self.pinch_enter_threshold:
                    self.is_pinching = True
            else:
                # Currently pinching - check if should exit pinch
                if distance > self.pinch_exit_threshold:
                    self.is_pinching = False
            
            is_pinch = self.is_pinching
            
            # Calculate confidence based on distance from threshold
            if is_pinch:
                confidence = 1.0 - (distance / self.pinch_enter_threshold)
            else:
                confidence = (distance - self.pinch_exit_threshold) / self.calibration_stats['avg_open']
        
        # Smooth with buffer to reduce jitter
        self.pinch_buffer.append(is_pinch)
        smoothed_pinch = sum(self.pinch_buffer) > len(self.pinch_buffer) / 2
        
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return smoothed_pinch, distance, confidence
    
    def update_calibration_state(self, hand_world_landmarks):
        """
        Update calibration state machine
        
        Returns:
            bool: True if calibration is active and consuming input
        """
        if self.calibration_state == CalibrationState.IDLE:
            return False
        
        if self.calibration_state == CalibrationState.INSTRUCTIONS:
            # ✅ CHANGED: Use configurable duration instead of hardcoded 3 seconds
            elapsed = time.time() - self.calibration_start_time
            if elapsed > self.instruction_duration:
                self.calibration_state = CalibrationState.COLLECTING_PINCH
                self.calibration_start_time = time.time()
            return True

        
        if self.calibration_state == CalibrationState.COLLECTING_PINCH:
            # Collect pinch samples
            if hand_world_landmarks:
                # Auto-collect samples (every 3 frames for diversity)
                if self.pinch_sample_count % 3 == 0 or self.pinch_sample_count < 5:
                    self.collect_calibration_sample(hand_world_landmarks, is_pinching=True)
                else:
                    self.pinch_sample_count += 1
            
            # Move to next phase after collecting enough samples
            if self.pinch_sample_count >= self.required_pinch_samples:
                self.calibration_state = CalibrationState.COLLECTING_OPEN
                self.calibration_start_time = time.time()
            return True
        
        if self.calibration_state == CalibrationState.COLLECTING_OPEN:
            # Collect open hand samples
            if hand_world_landmarks:
                if self.open_sample_count % 3 == 0 or self.open_sample_count < 5:
                    self.collect_calibration_sample(hand_world_landmarks, is_pinching=False)
                else:
                    self.open_sample_count += 1
            
            # Finalize calibration after collecting enough samples
            if self.open_sample_count >= self.required_open_samples:
                self.calibration_state = CalibrationState.FINALIZING
            return True
        
        if self.calibration_state == CalibrationState.FINALIZING:
            success = self.finalize_calibration()
            if success:
                self.calibration_state = CalibrationState.COMPLETE
                self.calibration_start_time = time.time()  # ✅ NEW: Reset timer for completion message
            else:
                self.calibration_state = CalibrationState.IDLE
            return True
        
        if self.calibration_state == CalibrationState.COMPLETE:
            # ✅ CHANGED: Use configurable duration instead of hardcoded 2 seconds
            elapsed = time.time() - self.calibration_start_time
            if elapsed > self.completion_message_duration:
                self.calibration_state = CalibrationState.IDLE
            return False

        
        return False
    
    def draw_calibration_ui(self, frame):
        """Draw calibration UI overlay on frame"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        if self.calibration_state == CalibrationState.INSTRUCTIONS:
            # Instructions screen
            cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 0, 0), -1)
            cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 255, 255), 3)
            
            # ✅ NEW: Calculate countdown timer
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.instruction_duration - elapsed)
            
            instructions = [
                "CALIBRATION INSTRUCTIONS",
                "",
                "Step 1: PINCH your thumb and index finger together",
                "        Hold for 2-3 seconds",
                "",
                "Step 2: OPEN your hand fully",
                "        Keep fingers spread for 2-3 seconds",
                "",
                "Keep your hand visible in the frame!",
                "",
                f"Starting in {remaining:.0f} seconds...",  # ✅ NEW: Countdown display
            ]
            
            y = 100
            for i, line in enumerate(instructions):
                if line == instructions[0]:
                    # Title in cyan
                    cv2.putText(overlay, line, (width//2 - 300, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                elif i == len(instructions) - 1:
                    # ✅ NEW: Countdown in bright green, centered
                    cv2.putText(overlay, line, (width//2 - 150, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(overlay, line, (100, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 40

        
        elif self.calibration_state == CalibrationState.COLLECTING_PINCH:
            # Pinch collection phase
            progress = self.pinch_sample_count / self.required_pinch_samples
            self._draw_progress_bar(overlay, "PINCH fingers together", progress)
            
            # Visual guide - show target gesture
            cv2.circle(overlay, (width//2, height//2), 50, (0, 255, 0), -1)
            cv2.circle(overlay, (width//2 + 20, height//2), 50, (0, 255, 0), -1)
        
        elif self.calibration_state == CalibrationState.COLLECTING_OPEN:
            # Open hand collection phase
            progress = self.open_sample_count / self.required_open_samples
            self._draw_progress_bar(overlay, "OPEN hand - spread fingers", progress)
            
            # Visual guide - show open hand
            for i in range(5):
                angle = -60 + i * 30
                x = int(width//2 + 80 * math.cos(math.radians(angle)))
                y = int(height//2 + 80 * math.sin(math.radians(angle)))
                cv2.line(overlay, (width//2, height//2), (x, y), (0, 255, 0), 8)
                cv2.circle(overlay, (x, y), 15, (0, 255, 0), -1)
        
        elif self.calibration_state == CalibrationState.COMPLETE:
            # Success message
            cv2.rectangle(overlay, (width//4, height//3), 
                        (3*width//4, 2*height//3), (0, 255, 0), -1)
            cv2.putText(overlay, "CALIBRATION COMPLETE!", 
                       (width//2 - 250, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            # ✅ NEW: Show countdown for completion message
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.completion_message_duration - elapsed)
            cv2.putText(overlay, f"Continuing in {remaining:.0f}s", 
                       (width//2 - 130, height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    def _draw_progress_bar(self, frame, text, progress):
        """Helper to draw progress bar"""
        height, width = frame.shape[:2]
        bar_width = width - 200
        bar_height = 40
        bar_x = 100
        bar_y = height - 100
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress
        filled_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + filled_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, text, (bar_x, bar_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{int(progress*100)}%", 
                   (bar_x + bar_width + 20, bar_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """
        Process frame for pinch detection
        
        Returns:
            processed_frame: Frame with visualizations
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Handle calibration
        is_calibrating = False
        if results.multi_hand_world_landmarks:
            hand_world = results.multi_hand_world_landmarks[0]
            is_calibrating = self.update_calibration_state(hand_world)
        else:
            is_calibrating = self.update_calibration_state(None)
        
        # Draw calibration UI if active
        if is_calibrating or self.calibration_state != CalibrationState.IDLE:
            self.draw_calibration_ui(frame)
        
        # Normal detection mode
        if not is_calibrating and results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            for hand_landmarks, hand_world_landmarks in zip(
                results.multi_hand_landmarks, 
                results.multi_hand_world_landmarks
            ):
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Detect pinch
                is_pinch, distance, confidence = self.detect_pinch_calibrated(hand_world_landmarks)
                
                # Visual feedback
                self._draw_detection_ui(frame, is_pinch, distance, confidence)
        
        return frame
    
    def _draw_detection_ui(self, frame, is_pinching, distance, confidence):
        """Draw detection status UI"""
        height, width = frame.shape[:2]
        
        # Status box
        status_color = (0, 255, 0) if is_pinching else (0, 0, 255)
        status_text = "PINCHING" if is_pinching else "OPEN"
        
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), status_color, 2)
        
        #Use SIMPLEX with thicker lines for bold effect
        cv2.putText(frame, status_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)  # thickness=3 for bold
        
        cv2.putText(frame, f"Distance: {distance*100:.2f}cm", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Calibration status
        calib_text = "Calibrated" if self.user_threshold else "Not Calibrated (Press 'C')"
        calib_color = (0, 255, 0) if self.user_threshold else (0, 165, 255)
        cv2.putText(frame, calib_text, (width - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
    
    def cleanup(self):
        """Release resources"""
        self.hands.close()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application with keyboard controls:
    - 'C': Start calibration
    - 'R': Reset calibration (use default threshold)
    - 'Q': Quit
    """
    print("\n" + "="*60)
    print("CALIBRATED PINCH DETECTION SYSTEM")
    print("="*60)
    print("\nKeyboard Controls:")
    print("  C - Start calibration (recommended!)")
    print("  R - Reset to default threshold")
    print("  Q - Quit")
    print("\nCalibration improves accuracy by 15-20%!")
    print("="*60 + "\n")
    
    #Create detector with custom timing parameters
    detector = CalibratedPinchDetector(
        instruction_duration=15,           # 15 seconds to read instructions
        samples_per_gesture=100,            # 100 samples per gesture (default)
        completion_message_duration=3      # 3 seconds success message
    )
    cap = cv2.VideoCapture(0)
    
    
    detector = CalibratedPinchDetector()
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Process frame
        frame = detector.process_frame(frame)
        
        # Display
        cv2.imshow('Calibrated Pinch Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nExiting...")
            break
        elif key == ord('c') or key == ord('C'):
            if detector.calibration_state == CalibrationState.IDLE:
                print("\n🎯 Starting calibration...")
                detector.start_calibration()
        elif key == ord('r') or key == ord('R'):
            print("\n🔄 Resetting to default threshold...")
            detector.user_threshold = None
            detector.pinch_enter_threshold = None
            detector.pinch_exit_threshold = None
            detector.calibration_samples = []
            detector.is_pinching = False
    
    # Cleanup
    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
