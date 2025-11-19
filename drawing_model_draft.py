import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

class AirCanvasApp:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Canvas setup
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.drawing_points = deque(maxlen=512)
        self.undo_stack = []
        self.redo_stack = []
        
        # State variables
        self.is_drawing = False
        self.clear_timer = 0
        self.clear_hold_time = 30  # frames (~1 second at 30fps)
        
        # Color palette and brush
        self.color_palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
        self.color_index = 0
        self.draw_color = self.color_palette[self.color_index]
        
        # Shake detection params
        self.prev_fingertip_x = None
        self.shake_threshold = 40  # pixels
        self.last_shake_time = 0
        self.shake_delay = 0.5  # seconds
        
    def calculate_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two landmarks"""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )
    
    def detect_pinch(self, hand_landmarks):
        """Detect thumb-index finger pinch"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05  # threshold
    
    def detect_open_palm(self, hand_landmarks):
        """Detect all 5 fingers raised except thumb bent"""
        fingers_up = 0
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers_up += 1
        
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        thumb_up = thumb_tip.x < thumb_base.x  # Left or right hand adjusts this
        
        return fingers_up >= 4 and not thumb_up  # Thumb down for open palm clear
    
    def detect_undo(self, hand_landmarks):
        """Detect thumb+pinky up, all other fingers down for undo"""
        thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
        pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y
        middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y
        ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y
        return thumb_up and pinky_up and not index_up and not middle_up and not ring_up
    
    def detect_redo(self, hand_landmarks):
        """Detect index+pinky up, all other fingers down for redo"""
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y
        pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y
        thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
        middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y
        ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y
        return index_up and pinky_up and not thumb_up and not middle_up and not ring_up
    
    def detect_shake_right(self, current_x):
        """Detect fast rightward movement of fingertip to switch color"""
        if self.prev_fingertip_x is None:
            self.prev_fingertip_x = current_x
            self.last_shake_time = 0
            return False
        dx = current_x - self.prev_fingertip_x
        dt = time.time() - self.last_shake_time
        if dx > self.shake_threshold and dt > self.shake_delay:
            self.last_shake_time = time.time()
            self.prev_fingertip_x = current_x
            return True
        self.prev_fingertip_x = current_x
        return False
    
    def undo_last_stroke(self):
        """Remove last stroke from drawing points and save it for redo"""
        if not self.drawing_points:
            return
        try:
            last_none = len(self.drawing_points) - 1 - list(self.drawing_points)[::-1].index(None)
        except ValueError:
            last_none = -1
        stroke = []
        # Collect points from last None to end
        while len(self.drawing_points) > last_none + 1:
            stroke.append(self.drawing_points.pop())
        if stroke:
            self.undo_stack.append(stroke[::-1])  # Save for redo
    
    def redo_last_stroke(self):
        """Re-add last undone stroke to drawing points"""
        if not self.undo_stack:
            return
        stroke = self.undo_stack.pop()
        for point in stroke:
            self.drawing_points.append(point)
    
    def get_fingertip_position(self, hand_landmarks, frame_shape):
        """Convert normalized landmark to pixel position"""
        h, w, _ = frame_shape
        index_tip = hand_landmarks.landmark[8]
        return (int(index_tip.x * w), int(index_tip.y * h))
    
    def calculate_brush_thickness(self, hand_landmarks):
        """Map pinch distance to brush thickness"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        dist = self.calculate_distance(thumb_tip, index_tip)
        min_thickness = 1
        max_thickness = 20
        dist = max(min(dist, 0.1), 0.01)
        thickness = int(min_thickness + (dist - 0.01) * (max_thickness - min_thickness) / 0.09)
        return thickness
    
    def draw_strokes(self):
        """Draw strokes on canvas from point deque"""
        for i in range(1, len(self.drawing_points)):
            if self.drawing_points[i-1] is None or self.drawing_points[i] is None:
                continue
            cv2.line(self.canvas, 
                     self.drawing_points[i-1], 
                     self.drawing_points[i], 
                     self.draw_color, 
                     self.brush_thickness)
    
    def draw_ui(self, frame):
        """Draw UI on frame including instructions and color palette"""
        cv2.putText(frame, "Pinch: Draw | Open Palm (1s): Clear | Shake Right: Color | Thumb+Pinky: Undo | Index+Pinky: Redo",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw color palette
        for i, col in enumerate(self.color_palette):
            cv2.rectangle(frame, (10 + i*50, 0), (60 + i*50, 30), col, -1)
        # Highlight active color
        cv2.rectangle(frame, (10 + self.color_index*50, 0), (60 + self.color_index*50, 30), (255, 255, 255), 2)
        # Clear progress bar
        if self.clear_timer > 0:
            progress = int((self.clear_timer / self.clear_hold_time) * 640)
            cv2.rectangle(frame, (0, 460), (progress, 480), (0, 255, 255), -1)
    
    def process_frame(self, frame):
        """Main frame processor with gesture detection and drawing logic"""
        frame = cv2.flip(frame, 1)  # Mirror for natural self view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        shake_detected = False
        undo_detected = False
        redo_detected = False
        pinch_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                pos = self.get_fingertip_position(hand_landmarks, frame.shape)
                
                # Detect shake right gesture to change color
                if self.detect_shake_right(pos[0]):
                    self.color_index = (self.color_index + 1) % len(self.color_palette)
                    self.draw_color = self.color_palette[self.color_index]
                    shake_detected = True
                
                # Detect undo gesture
                if self.detect_undo(hand_landmarks):
                    self.undo_last_stroke()
                    undo_detected = True
                
                # Detect redo gesture
                elif self.detect_redo(hand_landmarks):
                    self.redo_last_stroke()
                    redo_detected = True
                
                # Detect pinch to draw
                if self.detect_pinch(hand_landmarks):
                    self.brush_thickness = self.calculate_brush_thickness(hand_landmarks)
                    self.drawing_points.append(pos)
                    self.is_drawing = True
                    self.clear_timer = 0
                    pinch_detected = True
                else:
                    self.is_drawing = False
                    if len(self.drawing_points) > 0:
                        self.drawing_points.append(None)  # Break stroke if not drawing
                
                # Detect open palm to clear canvas
                if self.detect_open_palm(hand_landmarks):
                    self.clear_timer += 1
                    if self.clear_timer >= self.clear_hold_time:
                        self.canvas[:] = 0
                        self.drawing_points.clear()
                        self.undo_stack.clear()
                        self.redo_stack.clear()
                        self.clear_timer = 0
                        cv2.putText(frame, "CLEARED!", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    if not pinch_detected:
                        self.clear_timer = 0
        
        # Draw strokes on canvas
        self.draw_strokes()
        # Combine canvas with video frame
        combined = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
        
        # Show feedback text
        if shake_detected:
            cv2.putText(combined, "Color Changed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.draw_color, 2)
        if undo_detected:
            cv2.putText(combined, "UNDO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if redo_detected:
            cv2.putText(combined, "REDO!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw UI elements
        self.draw_ui(combined)
        
        return combined
    
    def run(self):
        """Open camera and run main loop"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output = self.process_frame(frame)
            cv2.imshow('Air Canvas', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Run the app
if __name__ == "__main__":
    app = AirCanvasApp()
    app.run()
