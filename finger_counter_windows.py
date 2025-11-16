import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
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
    thumb_raised = 1 if landmarks[4].x < landmarks[3].x else 0

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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Count fingers
            fingers_count, thumb_up = count_fingers(landmarks)

            # Display finger count
            cv2.putText(
                frame,
                f"Fingers: {fingers_count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

            # Display thumb status
            if thumb_up:
                cv2.putText(
                    frame,
                    "Thumb: UP",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
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
        cv2.imshow("Finger Counter", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()