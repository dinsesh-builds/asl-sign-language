import os
import cv2

# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = './data'
NUM_CLASSES = 36          # 26 letters + 10 digits
IMAGES_PER_CLASS = 300

# Create main data directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Character mapping (for display only)
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9'
]

# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(0)

for class_id in range(NUM_CLASSES):

    class_folder = os.path.join(DATA_DIR, str(class_id))

    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    print(f"\nCollecting images for class {class_id} ({class_names[class_id]})")

    # Wait for user to press Q to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        cv2.putText(display_frame,
                    f"Press Q to start capturing: {class_names[class_id]}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        cv2.imshow("Collect Images", display_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start Capturing
    count = 0

    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display,
                    f"Capturing {class_names[class_id]}: {count}/{IMAGES_PER_CLASS}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        cv2.imshow("Collect Images", display)
        cv2.imwrite(
            os.path.join(class_folder, f"{count}.jpg"),
            frame
        )

        count += 1

        if cv2.waitKey(25) & 0xFF == ord('x'):
            print("Stopped manually.")
            break

    print(f"Done collecting for {class_names[class_id]}")

cap.release()
cv2.destroyAllWindows()

print("\n✅ DATA COLLECTION FINISHED")