import os
import pickle
import cv2

import mediapipe as mp
mp_hands_module = getattr(mp, 'solutions')
mp_hands_module = getattr(mp_hands_module, 'hands')

hands = mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATA_DIR = './data'
data     = []
labels   = []

if not os.path.exists(DATA_DIR):
    print(f"ERROR: '{DATA_DIR}' does not exist!")
    exit()

def extract_features(hand_landmarks):
    """
    Extract 63 scale-invariant features from 21 landmarks:
      - 42 normalised x,y coords (divided by hand span → scale invariant)
      - 21 distances from each landmark to the wrist (landmark 0)
    
    Dividing by (x_max - x_min) makes features identical whether
    your hand is 20cm or 60cm from the camera.
    """
    lms = hand_landmarks.landmark

    x_ = [lm.x for lm in lms]
    y_ = [lm.y for lm in lms]

    x_min, x_max = min(x_), max(x_)
    y_min, y_max = min(y_), max(y_)

    x_span = (x_max - x_min) or 1e-6
    y_span = (y_max - y_min) or 1e-6

    # 42 normalised position features (0.0 – 1.0)
    features = []
    for lm in lms:
        features.append((lm.x - x_min) / x_span)
        features.append((lm.y - y_min) / y_span)

    # 21 distance features: each landmark → wrist (landmark 0)
    wrist = lms[0]
    for lm in lms:
        dx = (lm.x - wrist.x) / x_span
        dy = (lm.y - wrist.y) / y_span
        features.append((dx**2 + dy**2) ** 0.5)

    return features   # length = 63

dir_list = sorted(os.listdir(DATA_DIR), key=lambda d: (0, int(d)) if d.isdigit() else (1, d))
print(f"Found {len(dir_list)} folders: {dir_list}\n")

for dir_ in dir_list:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"  Processing '{dir_}'...", end=" ", flush=True)
    ok = 0
    skip = 0

    for img_file in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_file))
        if img is None:
            continue

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            skip += 1
            continue

        features = extract_features(results.multi_hand_landmarks[0])

        if len(features) != 63:
            skip += 1
            continue

        data.append(features)
        labels.append(dir_)
        ok += 1

    print(f"{ok} captured, {skip} skipped")

if not data:
    print("\nFATAL: No data captured.")
else:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"\nSUCCESS: {len(data)} samples, {len(set(labels))} classes, 63 features each.")
    print(f"Distribution: { {l: labels.count(l) for l in sorted(set(labels), key=lambda d: int(d) if d.isdigit() else d)} }")