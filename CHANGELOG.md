# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Light and dark themes, keyboard focus states and reduced motion support.
- Stop camera control and an on-device privacy notice.
- Model confidence gating and an explicit fingerspelling-only scope note.

### Fixed
- Training and inference no longer disagree about mirroring.
- Augmentation no longer leaks near duplicates into the test split.
- Browser fallback label order matches the ONNX class order.
- Saved training images no longer contain the capture overlay.
