# Security

This project runs in your browser or on your machine. Camera frames are processed locally and are not uploaded.

The browser demo loads MediaPipe and ONNX Runtime from a CDN. Those scripts come from jsdelivr. If your environment blocks third party scripts, use the Python path instead.

Do not commit real credentials, private data or personal recordings. `model.p`, `model.onnx` and `labels.json` are model files, not secrets.

Report security problems privately through GitHub's reporting channel or message the repository owner. Do not attach private data to a public issue.
