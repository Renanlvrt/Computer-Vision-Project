import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def count_fingers(landmarks, handedness):
    """
    Count raised fingers from hand landmarks.
    Returns: (fingers_count, thumb_is_up)
    """

    # Finger tip indices (MediaPipe hand landmarks)
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    fingers_raised = 0

    # Check each finger
    for tip in finger_tips:
        # If tip.y < pip_joint.y → finger is raised
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers_raised += 1

    # Check thumb separately (different axis)
    if handedness == "Right":
        thumb_raised = 1 if landmarks[4].x < landmarks[3].x else 0
    else:  # Left hand: reverse comparison
        thumb_raised = 1 if landmarks[4].x > landmarks[3].x else 0
        
    return fingers_raised, thumb_raised


def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_frame)
        
        total_fingers = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for (hand_landmarks, hand_handedness) in zip(results.multi_hand_landmarks, results.multi_handedness):
                #hand_landmarks = results.multi_hand_landmarks[0]

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark
                label = hand_handedness.classification[0].label  # "Left" or "Right"

                # Count fingers
                fingers_count, thumb_up = count_fingers(landmarks, label)
                total_fingers += fingers_count + thumb_up

                y_offset = 100
                cv2.putText(
                    frame,
                    f"{label} Hand: {fingers_count + thumb_up} fingers",
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (200, 100, 255),
                    2
                )
                if thumb_up:
                    cv2.putText(
                        frame,
                        "Thumb: UP",
                        (350, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (100, 255, 255),
                        2
                    )
                y_offset += 60  # Move the next hand display down

            # Display the TOTAL after the loop, so only one total appears:
            cv2.putText(
                frame,
                f"Total Fingers: {total_fingers}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
                
        else:
            cv2.putText(
                frame,
                "Show your hand to camera",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        

        # Display window
        cv2.imshow("Finger Counter", frame,)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()