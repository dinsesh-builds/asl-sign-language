# Contributing

Issues and patches are welcome. Please keep the scope honest: this is fingerspelling practice, not an ASL interpreter.

## Development

Install the packages in `requirements.txt`, then run `python inference_classifier.py` for the desktop demo or open `asl_live.html` for the browser demo.

## Data

Collected data should be consented and kept local. Never upload camera frames. If you change feature extraction, change `create_dataset.py`, `inference_classifier.py` and `asl_live.html` together so all three stay identical.

## Changes

Add a line to `CHANGELOG.md` under Unreleased. State any accuracy number with the split and dataset it came from.
