# ASL sign language recognition

Reads 36 American Sign Language signs (A to Z, 0 to 9) from a webcam feed. Two ways to run it:

- `inference_classifier.py` opens a live desktop window and builds a sentence as you sign.
- `asl_live.html` runs the exported model in the browser. The page loads `model.onnx` automatically, with a file picker as fallback.

## Live demo

https://dinsesh-builds.github.io/asl-sign-language/

## Run on desktop

Use Python 3.9 or newer. From the repository root:

```bash
pip install -r requirements.txt
python inference_classifier.py
```

On Windows, text to speech uses SAPI when available. Elsewhere it falls back to `pyttsx3`, which is optional:

```bash
pip install pyttsx3
```

Keys in the live window: `space` ends a word, `s` speaks the sentence, `t` toggles speech, `f` toggles letter feedback, `backspace` deletes a character, `c` clears, `q` quits.

## How it works

1. `collect_imgs.py` captures hand images for each of the 36 classes.
2. `create_dataset.py` converts hand landmarks into 63 scale invariant features (normalised x and y for 21 landmarks, plus the distance of each landmark to the wrist) and writes `data.pickle`.
3. `train_classifier.py` augments the data, makes a stratified 80/20 split, searches SVM parameters with 5 fold cross validation, then writes `model.p` and prints a classification report.
4. `export_model.py` writes `model.onnx` for the browser.
5. `inference_classifier.py` reads the webcam and uses a 7 frame majority buffer before appending a letter.

## Verify the result

```bash
python train_classifier.py
```

The script prints test accuracy and a per class report. This README does not claim a fixed accuracy number. Run it on your own data to get yours.

## Limits

- One hand at a time, detection confidence 0.75, tracking confidence 0.65.
- Trained on a controlled dataset. Lighting, camera angle, and hand size change the result.
- The browser path needs ONNX runtime and MediaPipe scripts from a CDN.
- This is a learning and prototyping tool, not a production accessibility product.

## Files

| File | Role |
| --- | --- |
| `collect_imgs.py` | Capture images per sign |
| `create_dataset.py` | Landmarks to features, writes `data.pickle` |
| `train_classifier.py` | Trains the SVM, writes `model.p`, prints metrics |
| `export_model.py` | Writes `model.onnx` for the browser |
| `inference_classifier.py` | Webcam demo with sentence building and speech |
| `asl_live.html` | Browser demo |
| `labels.json` | Label map |

## License

MIT. See [LICENSE](LICENSE).
