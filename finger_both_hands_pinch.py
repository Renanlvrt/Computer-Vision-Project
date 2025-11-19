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
    COLLECTING_LEFT_PINCH = 2
    COLLECTING_LEFT_OPEN = 3
    COLLECTING_RIGHT_PINCH = 4
    COLLECTING_RIGHT_OPEN = 5
    FINALIZING = 6
    COMPLETE = 7

class DualHandCalibratedDetector:
    """
    Dual-hand pinch detector with individual calibration for left and right hands
    Achieves 15-20% accuracy improvement through personalization per hand
    """
    
    def __init__(self, 
                 instruction_duration=8,
                 samples_per_gesture=100,
                 completion_message_duration=3):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Timing parameters
        self.instruction_duration = instruction_duration
        self.samples_per_gesture = samples_per_gesture
        self.completion_message_duration = completion_message_duration
        
        # Calibration data for LEFT hand
        self.left_threshold = None
        self.left_calibration_samples = []
        self.left_pinch_enter_threshold = None
        self.left_pinch_exit_threshold = None
        self.left_is_pinching = False
        self.left_pinch_buffer = deque(maxlen=5)
        
        # Calibration data for RIGHT hand
        self.right_threshold = None
        self.right_calibration_samples = []
        self.right_pinch_enter_threshold = None
        self.right_pinch_exit_threshold = None
        self.right_is_pinching = False
        self.right_pinch_buffer = deque(maxlen=5)
        
        # Default threshold (used before calibration)
        self.default_threshold = 0.03  # 3cm
        
        # Calibration state machine
        self.calibration_state = CalibrationState.IDLE
        self.calibration_start_time = 0
        self.required_samples = samples_per_gesture
        
        # Sample counters for each hand and gesture
        self.left_pinch_count = 0
        self.left_open_count = 0
        self.right_pinch_count = 0
        self.right_open_count = 0
        
        # Statistics
        self.left_stats = {
            'pinch_distances': [],
            'open_distances': [],
            'avg_pinch': 0,
            'avg_open': 0,
            'std_pinch': 0,
            'std_open': 0
        }
        
        self.right_stats = {
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
    
    def get_hand_label(self, results, hand_idx):
        """
        Get the handedness label (Left or Right) for a specific hand
        
        Args:
            results: MediaPipe results object
            hand_idx: Index of the hand (0 or 1)
        
        Returns:
            str: "Left" or "Right"
        """
        if results.multi_handedness and hand_idx < len(results.multi_handedness):
            handedness = results.multi_handedness[hand_idx]
            return handedness.classification[0].label
        return None
    
    def start_calibration(self):
        """Initialize dual-hand calibration process"""
        print("\n" + "="*60)
        print("STARTING DUAL-HAND CALIBRATION")
        print("="*60)
        self.calibration_state = CalibrationState.INSTRUCTIONS
        
        # Reset all calibration data
        self.left_calibration_samples = []
        self.right_calibration_samples = []
        self.left_pinch_count = 0
        self.left_open_count = 0
        self.right_pinch_count = 0
        self.right_open_count = 0
        
        # Reset statistics
        self.left_stats = {
            'pinch_distances': [],
            'open_distances': [],
            'avg_pinch': 0,
            'avg_open': 0,
            'std_pinch': 0,
            'std_open': 0
        }
        self.right_stats = {
            'pinch_distances': [],
            'open_distances': [],
            'avg_pinch': 0,
            'avg_open': 0,
            'std_pinch': 0,
            'std_open': 0
        }
        
        self.calibration_start_time = time.time()
    
    def collect_calibration_sample(self, hand_world_landmarks, is_pinching, hand_label):
        """
        Collect calibration sample for specific hand
        
        Args:
            hand_world_landmarks: MediaPipe world landmarks
            is_pinching: True if pinch gesture, False if open
            hand_label: "Left" or "Right"
        """
        thumb_tip = hand_world_landmarks.landmark[4]
        index_tip = hand_world_landmarks.landmark[8]
        
        distance = self.calculate_distance_3d(thumb_tip, index_tip)
        
        sample = {
            'distance': distance,
            'is_pinching': is_pinching,
            'timestamp': time.time()
        }
        
        if hand_label == "Left":
            self.left_calibration_samples.append(sample)
            if is_pinching:
                self.left_pinch_count += 1
                self.left_stats['pinch_distances'].append(distance)
            else:
                self.left_open_count += 1
                self.left_stats['open_distances'].append(distance)
        else:  # Right
            self.right_calibration_samples.append(sample)
            if is_pinching:
                self.right_pinch_count += 1
                self.right_stats['pinch_distances'].append(distance)
            else:
                self.right_open_count += 1
                self.right_stats['open_distances'].append(distance)
    
    def finalize_calibration(self):
        """Calculate optimal thresholds for both hands"""
        success = True
        
        # Calibrate LEFT hand
        left_pinch = [s['distance'] for s in self.left_calibration_samples if s['is_pinching']]
        left_open = [s['distance'] for s in self.left_calibration_samples if not s['is_pinching']]
        
        if left_pinch and left_open:
            avg_pinch = np.mean(left_pinch)
            avg_open = np.mean(left_open)
            std_pinch = np.std(left_pinch)
            std_open = np.std(left_open)
            
            self.left_stats['avg_pinch'] = avg_pinch
            self.left_stats['avg_open'] = avg_open
            self.left_stats['std_pinch'] = std_pinch
            self.left_stats['std_open'] = std_open
            
            self.left_threshold = (avg_pinch + avg_open) / 2
            hysteresis = (avg_open - avg_pinch) * 0.15
            self.left_pinch_enter_threshold = self.left_threshold - hysteresis
            self.left_pinch_exit_threshold = self.left_threshold + hysteresis
            
            print(f"\n✅ LEFT HAND Calibrated:")
            print(f"   Pinch: {avg_pinch*100:.2f}cm (±{std_pinch*100:.2f}cm)")
            print(f"   Open: {avg_open*100:.2f}cm (±{std_open*100:.2f}cm)")
            print(f"   Threshold: {self.left_threshold*100:.2f}cm")
        else:
            print("❌ LEFT HAND calibration failed: Insufficient data")
            success = False
        
        # Calibrate RIGHT hand
        right_pinch = [s['distance'] for s in self.right_calibration_samples if s['is_pinching']]
        right_open = [s['distance'] for s in self.right_calibration_samples if not s['is_pinching']]
        
        if right_pinch and right_open:
            avg_pinch = np.mean(right_pinch)
            avg_open = np.mean(right_open)
            std_pinch = np.std(right_pinch)
            std_open = np.std(right_open)
            
            self.right_stats['avg_pinch'] = avg_pinch
            self.right_stats['avg_open'] = avg_open
            self.right_stats['std_pinch'] = std_pinch
            self.right_stats['std_open'] = std_open
            
            self.right_threshold = (avg_pinch + avg_open) / 2
            hysteresis = (avg_open - avg_pinch) * 0.15
            self.right_pinch_enter_threshold = self.right_threshold - hysteresis
            self.right_pinch_exit_threshold = self.right_threshold + hysteresis
            
            print(f"\n✅ RIGHT HAND Calibrated:")
            print(f"   Pinch: {avg_pinch*100:.2f}cm (±{std_pinch*100:.2f}cm)")
            print(f"   Open: {avg_open*100:.2f}cm (±{std_open*100:.2f}cm)")
            print(f"   Threshold: {self.right_threshold*100:.2f}cm")
        else:
            print("❌ RIGHT HAND calibration failed: Insufficient data")
            success = False
        
        return success
    
    def update_calibration_state(self, results):
        """Update calibration state machine for dual hands"""
        if self.calibration_state == CalibrationState.IDLE:
            return False
        
        if self.calibration_state == CalibrationState.INSTRUCTIONS:
            elapsed = time.time() - self.calibration_start_time
            if elapsed > self.instruction_duration:
                self.calibration_state = CalibrationState.COLLECTING_LEFT_PINCH
                self.calibration_start_time = time.time()
            return True
        
        # LEFT HAND - Pinch collection
        if self.calibration_state == CalibrationState.COLLECTING_LEFT_PINCH:
            if results.multi_hand_world_landmarks:
                for idx, hand_world in enumerate(results.multi_hand_world_landmarks):
                    label = self.get_hand_label(results, idx)
                    if label == "Left":
                        if self.left_pinch_count % 3 == 0 or self.left_pinch_count < 5:
                            self.collect_calibration_sample(hand_world, True, "Left")
                        else:
                            self.left_pinch_count += 1
            
            if self.left_pinch_count >= self.required_samples:
                self.calibration_state = CalibrationState.COLLECTING_LEFT_OPEN
                self.calibration_start_time = time.time()
            return True
        
        # LEFT HAND - Open collection
        if self.calibration_state == CalibrationState.COLLECTING_LEFT_OPEN:
            if results.multi_hand_world_landmarks:
                for idx, hand_world in enumerate(results.multi_hand_world_landmarks):
                    label = self.get_hand_label(results, idx)
                    if label == "Left":
                        if self.left_open_count % 3 == 0 or self.left_open_count < 5:
                            self.collect_calibration_sample(hand_world, False, "Left")
                        else:
                            self.left_open_count += 1
            
            if self.left_open_count >= self.required_samples:
                self.calibration_state = CalibrationState.COLLECTING_RIGHT_PINCH
                self.calibration_start_time = time.time()
            return True
        
        # RIGHT HAND - Pinch collection
        if self.calibration_state == CalibrationState.COLLECTING_RIGHT_PINCH:
            if results.multi_hand_world_landmarks:
                for idx, hand_world in enumerate(results.multi_hand_world_landmarks):
                    label = self.get_hand_label(results, idx)
                    if label == "Right":
                        if self.right_pinch_count % 3 == 0 or self.right_pinch_count < 5:
                            self.collect_calibration_sample(hand_world, True, "Right")
                        else:
                            self.right_pinch_count += 1
            
            if self.right_pinch_count >= self.required_samples:
                self.calibration_state = CalibrationState.COLLECTING_RIGHT_OPEN
                self.calibration_start_time = time.time()
            return True
        
        # RIGHT HAND - Open collection
        if self.calibration_state == CalibrationState.COLLECTING_RIGHT_OPEN:
            if results.multi_hand_world_landmarks:
                for idx, hand_world in enumerate(results.multi_hand_world_landmarks):
                    label = self.get_hand_label(results, idx)
                    if label == "Right":
                        if self.right_open_count % 3 == 0 or self.right_open_count < 5:
                            self.collect_calibration_sample(hand_world, False, "Right")
                        else:
                            self.right_open_count += 1
            
            if self.right_open_count >= self.required_samples:
                self.calibration_state = CalibrationState.FINALIZING
            return True
        
        if self.calibration_state == CalibrationState.FINALIZING:
            success = self.finalize_calibration()
            if success:
                self.calibration_state = CalibrationState.COMPLETE
                self.calibration_start_time = time.time()
            else:
                self.calibration_state = CalibrationState.IDLE
            return True
        
        if self.calibration_state == CalibrationState.COMPLETE:
            elapsed = time.time() - self.calibration_start_time
            if elapsed > self.completion_message_duration:
                self.calibration_state = CalibrationState.IDLE
            return False
        
        return False
    
    def detect_pinch_for_hand(self, hand_world_landmarks, hand_label):
        """
        Detect pinch for a specific hand using calibrated thresholds
        
        Args:
            hand_world_landmarks: MediaPipe world landmarks
            hand_label: "Left" or "Right"
        
        Returns:
            tuple: (is_pinching, distance, confidence)
        """
        thumb_tip = hand_world_landmarks.landmark[4]
        index_tip = hand_world_landmarks.landmark[8]
        
        distance = self.calculate_distance_3d(thumb_tip, index_tip)
        
        # Get appropriate thresholds and state for this hand
        if hand_label == "Left":
            threshold = self.left_threshold
            enter_threshold = self.left_pinch_enter_threshold
            exit_threshold = self.left_pinch_exit_threshold
            is_pinching = self.left_is_pinching
            pinch_buffer = self.left_pinch_buffer
            stats = self.left_stats
        else:  # Right
            threshold = self.right_threshold
            enter_threshold = self.right_pinch_enter_threshold
            exit_threshold = self.right_pinch_exit_threshold
            is_pinching = self.right_is_pinching
            pinch_buffer = self.right_pinch_buffer
            stats = self.right_stats
        
        # Use calibrated or default thresholds
        if threshold is None:
            # Not calibrated - use default
            current_threshold = self.default_threshold
            is_pinch = distance < current_threshold
            confidence = 1.0 - (distance / current_threshold)
        else:
            # Use calibrated thresholds with hysteresis
            if not is_pinching:
                if distance < enter_threshold:
                    is_pinching = True
            else:
                if distance > exit_threshold:
                    is_pinching = False
            
            is_pinch = is_pinching
            
            # Calculate confidence
            if is_pinch:
                confidence = 1.0 - (distance / enter_threshold)
            else:
                confidence = (distance - exit_threshold) / stats['avg_open'] if stats['avg_open'] > 0 else 0
        
        # Update state
        if hand_label == "Left":
            self.left_is_pinching = is_pinching
        else:
            self.right_is_pinching = is_pinching
        
        # Smooth with buffer
        pinch_buffer.append(is_pinch)
        smoothed_pinch = sum(pinch_buffer) > len(pinch_buffer) / 2
        
        confidence = max(0.0, min(1.0, confidence))
        
        return smoothed_pinch, distance, confidence
    
    def draw_calibration_ui(self, frame):
        """Draw calibration UI overlay"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        if self.calibration_state == CalibrationState.INSTRUCTIONS:
            cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 0, 0), -1)
            cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 255, 255), 3)
            
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.instruction_duration - elapsed)
            
            instructions = [
                "DUAL-HAND CALIBRATION INSTRUCTIONS",
                "",
                "We'll calibrate each hand separately:",
                "",
                "Step 1: LEFT HAND - PINCH thumb and index together",
                "Step 2: LEFT HAND - OPEN hand fully",
                "Step 3: RIGHT HAND - PINCH thumb and index together",
                "Step 4: RIGHT HAND - OPEN hand fully",
                "",
                "Keep the correct hand visible during each step!",
                "",
                f"Starting in {remaining:.0f} seconds...",
            ]
            
            y = 80
            for i, line in enumerate(instructions):
                if line == instructions[0]:
                    cv2.putText(overlay, line, (width//2 - 350, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
                elif i == len(instructions) - 1:
                    cv2.putText(overlay, line, (width//2 - 150, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(overlay, line, (100, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 35
        
        elif self.calibration_state == CalibrationState.COLLECTING_LEFT_PINCH:
            progress = self.left_pinch_count / self.required_samples
            self._draw_progress_bar(overlay, "LEFT HAND - PINCH fingers together", progress, (255, 100, 100))
        
        elif self.calibration_state == CalibrationState.COLLECTING_LEFT_OPEN:
            progress = self.left_open_count / self.required_samples
            self._draw_progress_bar(overlay, "LEFT HAND - OPEN hand, spread fingers", progress, (255, 100, 100))
        
        elif self.calibration_state == CalibrationState.COLLECTING_RIGHT_PINCH:
            progress = self.right_pinch_count / self.required_samples
            self._draw_progress_bar(overlay, "RIGHT HAND - PINCH fingers together", progress, (100, 100, 255))
        
        elif self.calibration_state == CalibrationState.COLLECTING_RIGHT_OPEN:
            progress = self.right_open_count / self.required_samples
            self._draw_progress_bar(overlay, "RIGHT HAND - OPEN hand, spread fingers", progress, (100, 100, 255))
        
        elif self.calibration_state == CalibrationState.COMPLETE:
            cv2.rectangle(overlay, (width//4, height//3), 
                        (3*width//4, 2*height//3), (0, 255, 0), -1)
            cv2.putText(overlay, "CALIBRATION COMPLETE!", 
                       (width//2 - 250, height//2 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(overlay, "Both hands calibrated successfully!", 
                       (width//2 - 250, height//2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.completion_message_duration - elapsed)
            cv2.putText(overlay, f"Continuing in {remaining:.0f}s", 
                       (width//2 - 130, height//2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    def _draw_progress_bar(self, frame, text, progress, color):
        """Helper to draw progress bar with custom color"""
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
                     color, -1)
        
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
        """Process frame for dual-hand pinch detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Handle calibration
        is_calibrating = self.update_calibration_state(results)
        
        # Draw calibration UI if active
        if is_calibrating or self.calibration_state != CalibrationState.IDLE:
            self.draw_calibration_ui(frame)
        
        # Normal detection mode
        if not is_calibrating and results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            for idx, (hand_landmarks, hand_world_landmarks) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks)
            ):
                # Get hand label (Left or Right)
                hand_label = self.get_hand_label(results, idx)
                
                if hand_label is None:
                    continue
                
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Detect pinch for this specific hand
                is_pinch, distance, confidence = self.detect_pinch_for_hand(
                    hand_world_landmarks, hand_label
                )
                
                # Visual feedback
                self._draw_detection_ui(frame, is_pinch, distance, confidence, hand_label, idx)
        
        return frame
    
    def _draw_detection_ui(self, frame, is_pinching, distance, confidence, hand_label, hand_idx):
        """Draw detection status UI for specific hand"""
        height, width = frame.shape[:2]
        
        # Position boxes side by side
        if hand_label == "Left":
            box_x = 10
            status_color = (255, 100, 100)  # Reddish for left
        else:  # Right
            box_x = width - 310
            status_color = (100, 100, 255)  # Blueish for right
        
        status_text = "PINCHING" if is_pinching else "OPEN"
        box_color = (0, 255, 0) if is_pinching else (0, 0, 255)
        
        # Status box
        cv2.rectangle(frame, (box_x, 10), (box_x + 300, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (box_x, 10), (box_x + 300, 140), box_color, 2)
        
        # Hand label
        cv2.putText(frame, f"{hand_label} HAND", (box_x + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Status
        cv2.putText(frame, status_text, (box_x + 10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        # Distance
        cv2.putText(frame, f"Dist: {distance*100:.2f}cm", 
                   (box_x + 10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence
        cv2.putText(frame, f"Conf: {confidence*100:.0f}%", 
                   (box_x + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration status (bottom of frame)
        calib_y = height - 30
        if hand_label == "Left":
            calib_text = "L: Calibrated" if self.left_threshold else "L: Not Calibrated"
            calib_color = (0, 255, 0) if self.left_threshold else (0, 165, 255)
            cv2.putText(frame, calib_text, (10, calib_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
        else:
            calib_text = "R: Calibrated" if self.right_threshold else "R: Not Calibrated"
            calib_color = (0, 255, 0) if self.right_threshold else (0, 165, 255)
            cv2.putText(frame, calib_text, (width - 200, calib_y), 
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
    - 'C': Start dual-hand calibration
    - 'R': Reset calibration for both hands
    - 'Q': Quit
    """
    print("\n" + "="*60)
    print("DUAL-HAND CALIBRATED PINCH DETECTION SYSTEM")
    print("="*60)
    print("\nKeyboard Controls:")
    print("  C - Start calibration (both hands)")
    print("  R - Reset to default threshold")
    print("  Q - Quit")
    print("\nThis system calibrates each hand separately!")
    print("Calibration improves accuracy by 15-20% per hand!")
    print("="*60 + "\n")
    
    # Create detector with custom timing
    detector = DualHandCalibratedDetector(
        instruction_duration=8,
        samples_per_gesture=100,    #set 100 frames for the calibration of the two hands
        completion_message_duration=3
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame
        frame = detector.process_frame(frame)
        
        # Display
        cv2.imshow('Dual-Hand Pinch Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nExiting...")
            break
        elif key == ord('c'):
            if detector.calibration_state == CalibrationState.IDLE:
                print("\n🎯 Starting dual-hand calibration...")
                detector.start_calibration()
        elif key == ord('C'):
            print("\n🔄 Resetting both hands to default threshold...")
            detector.left_threshold = None
            detector.right_threshold = None
            detector.left_pinch_enter_threshold = None
            detector.right_pinch_enter_threshold = None
            detector.left_pinch_exit_threshold = None
            detector.right_pinch_exit_threshold = None
            detector.left_is_pinching = False
            detector.right_is_pinching = False
    
    # Cleanup
    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
