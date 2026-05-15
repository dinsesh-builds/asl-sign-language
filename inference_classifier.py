import pickle
import cv2
import numpy as np
import os
from spellchecker import SpellChecker
import time
import threading
import queue

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import mediapipe as mp
mp_hands_module = getattr(mp, 'solutions')
mp_hands        = getattr(mp_hands_module, 'hands')
mp_drawing      = getattr(mp_hands_module, 'drawing_utils')

# ---------------------------------------------------------------
# Unified 36-class label map
#   Folders 0-25  → A-Z
#   Folders 26-35 → 0-9
# ---------------------------------------------------------------
signs_dict = {}
for i in range(26):
    signs_dict[i] = chr(65 + i)   # 0→A … 25→Z
for i in range(10):
    signs_dict[26 + i] = str(i)   # 26→0 … 35→9

spell = SpellChecker()

BUFFER_SIZE          = 7
MIN_COUNT            = 4
REQUIRED_FRAMES      = 30
HAND_PAUSE_THRESHOLD = 1.0

# =====================================================================
# TTS — Windows SAPI (best quality) with pyttsx3 fallback
# =====================================================================
tts_queue       = queue.Queue()
tts_muted       = False
tts_speaking    = False
tts_engine_type = "none"

def _pick_best_pyttsx3_voice(engine):
    voices = engine.getProperty('voices')
    for kw in ['zira', 'hazel', 'susan', 'female', 'en-us', 'english']:
        for v in voices:
            if kw in v.id.lower() or kw in v.name.lower():
                return v.id
    return voices[0].id if voices else None

def _tts_worker():
    global tts_speaking, tts_engine_type
    sapi_ok = False
    py_engine = None

    try:
        import win32com.client
        sapi = win32com.client.Dispatch("SAPI.SpVoice")
        vs = sapi.GetVoices()
        for i in range(vs.Count):
            if any(k in vs.Item(i).GetDescription().lower() for k in ('zira','hazel','female')):
                sapi.Voice = vs.Item(i); break
        sapi.Rate = 1; sapi.Volume = 90
        sapi_ok = True; tts_engine_type = "sapi"
        print("TTS: Windows SAPI")
    except Exception:
        pass

    if not sapi_ok:
        try:
            import pyttsx3
            py_engine = pyttsx3.init()
            best = _pick_best_pyttsx3_voice(py_engine)
            if best: py_engine.setProperty('voice', best)
            py_engine.setProperty('rate', 155)
            py_engine.setProperty('volume', 1.0)
            tts_engine_type = "pyttsx3"
            print("TTS: pyttsx3")
        except Exception as e:
            tts_engine_type = "none"; print(f"TTS unavailable: {e}")

    while True:
        text = tts_queue.get()
        if text is None: break
        if tts_muted or not text.strip():
            tts_queue.task_done(); continue
        tts_speaking = True
        try:
            if sapi_ok:           sapi.Speak(text)
            elif py_engine:       py_engine.say(text); py_engine.runAndWait()
        except Exception as e:    print(f"TTS error: {e}")
        tts_speaking = False
        tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text, priority=False):
    if tts_muted or not text.strip(): return
    if priority:
        while not tts_queue.empty():
            try: tts_queue.get_nowait(); tts_queue.task_done()
            except queue.Empty: break
    tts_queue.put(text)

# =====================================================================
# FEATURE EXTRACTION — must exactly match create_dataset.py
# =====================================================================
def extract_features(hand_landmarks):
    """
    63 scale-invariant features:
      42 = normalised x,y per landmark (divided by hand bounding box)
      21 = distance from each landmark to wrist
    """
    lms   = hand_landmarks.landmark
    x_    = [lm.x for lm in lms]
    y_    = [lm.y for lm in lms]
    x_min, x_max = min(x_), max(x_)
    y_min, y_max = min(y_), max(y_)
    x_span = (x_max - x_min) or 1e-6
    y_span = (y_max - y_min) or 1e-6

    features = []
    for lm in lms:
        features.append((lm.x - x_min) / x_span)
        features.append((lm.y - y_min) / y_span)

    wrist = lms[0]
    for lm in lms:
        dx = (lm.x - wrist.x) / x_span
        dy = (lm.y - wrist.y) / y_span
        features.append((dx**2 + dy**2) ** 0.5)

    return features  # length 63

# =====================================================================
# LOAD MODEL
# =====================================================================
try:
    model_dict   = pickle.load(open('./model.p', 'rb'))
    model        = model_dict['model']
    N_FEATURES   = model_dict.get('n_features', 63)
    print(f"Model loaded. Expected features: {N_FEATURES}")
except Exception as e:
    print(f"FATAL: model.p not found — {e}"); exit()

# =====================================================================
# CAMERA + MEDIAPIPE
# =====================================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

Hands       = getattr(mp_hands, 'Hands')
DrawingSpec = getattr(mp_drawing, 'DrawingSpec')
hands = Hands(static_image_mode=False, max_num_hands=1,
              min_detection_confidence=0.75, min_tracking_confidence=0.65)

# =====================================================================
# APP STATE
# =====================================================================
current_sentence   = []
display_char       = ""
last_appended_char = ""
prediction_buffer  = []
prediction_count   = 0
letter_feedback    = False
confidence_pct     = 0   # live model confidence % to display

window_name = 'ASL Real-Time Keyboard'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

last_hand_time = time.time()
hand_visible   = False
flash_msg      = ""
flash_timer    = 0.0
FLASH_DURATION = 1.5

def set_flash(msg):
    global flash_msg, flash_timer
    flash_msg = msg; flash_timer = time.time()

# =====================================================================
# HELPERS
# =====================================================================
def auto_correct_last_word(char_list):
    full = "".join(char_list).rstrip()
    words = full.split()
    if not words: return char_list
    last = words[-1]
    if not last.isdigit() and len(last) > 1:
        corrected = spell.correction(last.lower())
        if corrected and corrected.upper() != last:
            words[-1] = corrected.upper()
            new = " ".join(words)
            if "".join(char_list).endswith(" "): new += " "
            return list(new)
    return char_list

def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                   (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)

def end_word(speak_word=True):
    global current_sentence, last_appended_char
    if current_sentence and current_sentence[-1] != " ":
        current_sentence = auto_correct_last_word(current_sentence)
        if speak_word:
            words = "".join(current_sentence).strip().split()
            if words: speak(words[-1])
        current_sentence.append(" ")
        last_appended_char = " "

# =====================================================================
# MAIN LOOP
# =====================================================================
while True:
    ret, frame = cap.read()
    if not ret: break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    frame     = cv2.flip(frame, 1)
    H, W, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)
    font      = cv2.FONT_HERSHEY_SIMPLEX

    # ---------- Header ----------
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,68), (18,18,18), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    mute_label = "MUTED" if tts_muted else ("SPEAKING..." if tts_speaking else "ON")
    mute_color = (60,60,210) if tts_muted else ((0,210,100) if tts_speaking else (80,180,80))

    cv2.putText(frame, "SPACE:End Word  S:Speak  BKSP:Delete  C:Clear  Q:Quit",
                (15,28), font, 0.50, (190,190,190), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"A-Z+0-9 AUTO | T:TTS[{mute_label}] | F:Feedback[{'ON' if letter_feedback else 'OFF'}] | ENGINE:{tts_engine_type.upper()}",
                (15,55), font, 0.50, mute_color, 1, cv2.LINE_AA)

    # ---------- Hand detection ----------
    detected_char = ""

    if results.multi_hand_landmarks:
        hand         = results.multi_hand_landmarks[0]
        hand_visible = True
        last_hand_time = time.time()

        HAND_CONNECTIONS = getattr(mp_hands, 'HAND_CONNECTIONS')
        mp_drawing.draw_landmarks(frame, hand, HAND_CONNECTIONS,
            DrawingSpec(color=(255,255,255), thickness=1, circle_radius=3),
            DrawingSpec(color=(0,200,255),   thickness=2))

        features = extract_features(hand)

        if len(features) == N_FEATURES:
            try:
                feat_arr   = np.asarray(features).reshape(1, -1)
                pred_label = model.predict(feat_arr)[0]
                pred       = int(pred_label)
                char       = signs_dict.get(pred, "")

                # Get model confidence (probability of top prediction)
                if hasattr(model, 'predict_proba'):
                    proba        = model.predict_proba(feat_arr)[0]
                    confidence_pct = int(max(proba) * 100)
                else:
                    confidence_pct = 0

                # Debug line — uncomment if something seems wrong:
                # print(f"pred={pred} char={char!r} conf={confidence_pct}%")

                if char:
                    prediction_buffer.append(char)
                    if len(prediction_buffer) > BUFFER_SIZE:
                        prediction_buffer.pop(0)

                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    buf_conf    = prediction_buffer.count(most_common)

                    if buf_conf >= MIN_COUNT:
                        detected_char = most_common

                        if detected_char == last_appended_char:
                            prediction_count = min(prediction_count+1, REQUIRED_FRAMES)
                        else:
                            prediction_count   = 1
                            last_appended_char = detected_char

                        if prediction_count >= REQUIRED_FRAMES:
                            current_sentence.append(detected_char)
                            prediction_count   = 0
                            last_appended_char = ""
                            if letter_feedback: speak(detected_char)
                    else:
                        prediction_count = max(0, prediction_count-1)

            except Exception as e:
                print(f"Prediction Error: {e}")
        else:
            print(f"Feature mismatch: got {len(features)}, expected {N_FEATURES}. Re-run create_dataset.py")

    else:
        hand_visible     = False
        detected_char    = ""
        confidence_pct   = 0
        prediction_count = max(0, prediction_count-2)

        if time.time() - last_hand_time > HAND_PAUSE_THRESHOLD:
            end_word(speak_word=True)
            last_hand_time = time.time()

    display_char = detected_char

    # ---------- Bottom panel ----------
    panel_y = H - 210
    pov = frame.copy()
    cv2.rectangle(pov, (0, panel_y), (W, H), (14,14,14), -1)
    cv2.addWeighted(pov, 0.65, frame, 0.35, 0, frame)

    if display_char:
        # Letter box
        box_size = 100
        bx = (W//2) - box_size//2
        by = panel_y + 10
        draw_rounded_rect(frame, (bx,by), (bx+box_size, by+box_size), (42,42,42), radius=12)
        t_size = cv2.getTextSize(display_char, font, 3.5, 5)[0]
        cv2.putText(frame, display_char,
                    (bx+(box_size-t_size[0])//2, by+(box_size+t_size[1])//2),
                    font, 3.5, (0,230,255), 5, cv2.LINE_AA)

        # Model confidence badge
        if confidence_pct > 0:
            conf_color = (0,200,80) if confidence_pct>=70 else (0,165,255) if confidence_pct>=45 else (0,60,200)
            cv2.putText(frame, f"{confidence_pct}% confident",
                        (bx + box_size + 15, by + 40), font, 0.6, conf_color, 2, cv2.LINE_AA)

        # HOLD progress bar
        bar_x, bar_y = (W//2)-120, panel_y+125
        bar_w, bar_h = 240, 14
        ratio       = prediction_count / REQUIRED_FRAMES
        prog_w      = int(ratio * bar_w)
        bar_color   = (0, int(200*ratio+100), int(255*(1-ratio)))

        cv2.rectangle(frame, (bar_x,bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x,bar_y), (bar_x+bar_w, bar_y+bar_h), (85,85,85), 1)
        if prog_w > 0:
            cv2.rectangle(frame, (bar_x,bar_y), (bar_x+prog_w, bar_y+bar_h), bar_color, -1)
        cv2.putText(frame, f"HOLD {int(ratio*100)}%",
                    (bar_x+bar_w+10, bar_y+12), font, 0.5, (170,170,170), 1, cv2.LINE_AA)
    else:
        msg    = "Show your hand..." if not hand_visible else "Stabilize gesture..."
        m_size = cv2.getTextSize(msg, font, 0.8, 1)[0]
        cv2.putText(frame, msg, ((W-m_size[0])//2, panel_y+70),
                    font, 0.8, (95,95,95), 1, cv2.LINE_AA)

    # ---------- Sentence ----------
    sentence_text    = "".join(current_sentence)
    disp_sentence    = ("..."+sentence_text[-35:]) if len(sentence_text)>35 else sentence_text
    s_size           = cv2.getTextSize(disp_sentence, font, 1.4, 2)[0]
    cv2.putText(frame, disp_sentence, (max(10,(W-s_size[0])//2), H-30),
                font, 1.4, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Sentence:", (15, H-55), font, 0.52, (130,130,130), 1, cv2.LINE_AA)

    # ---------- Flash ----------
    if flash_msg and (time.time()-flash_timer) < FLASH_DURATION:
        f_size = cv2.getTextSize(flash_msg, font, 1.0, 2)[0]
        fx = (W-f_size[0])//2; fy = H//2
        cv2.rectangle(frame, (fx-18,fy-38), (fx+f_size[0]+18, fy+12), (0,0,0), -1)
        cv2.rectangle(frame, (fx-18,fy-38), (fx+f_size[0]+18, fy+12), (0,180,100), 2)
        cv2.putText(frame, flash_msg, (fx,fy), font, 1.0, (0,230,150), 2, cv2.LINE_AA)
    else:
        flash_msg = ""

    cv2.imshow(window_name, frame)

    # ---------- Keys ----------
    key = cv2.waitKey(1) & 0xFF

    if   key == ord('q'): break
    elif key == 32:        end_word(speak_word=True);  set_flash("Word added")
    elif key == ord('s'):
        full = "".join(current_sentence).strip()
        if full: speak(full, priority=True); set_flash("Speaking sentence...")
    elif key == ord('t'):
        tts_muted = not tts_muted
        set_flash("TTS MUTED" if tts_muted else "TTS ON")
    elif key == ord('f'):
        letter_feedback = not letter_feedback
        set_flash(f"Letter feedback: {'ON' if letter_feedback else 'OFF'}")
    elif key == 8:
        if current_sentence: current_sentence.pop()
    elif key == ord('c'):
        current_sentence.clear(); display_char=""; prediction_count=0
        last_appended_char=""; prediction_buffer.clear(); set_flash("Cleared")

tts_queue.put(None)
cap.release()
cv2.destroyAllWindows()