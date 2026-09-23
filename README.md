# 🤟 Sign Language Translator

> Computer-vision and machine-learning prototype for recognizing predefined hand gestures and converting them into text and speech.

## Overview

This project explores real-time sign/gesture recognition using a webcam, OpenCV, and machine-learning techniques.

The original project was built as an experimental prototype. It is intentionally documented with its current scope and limitations rather than presenting it as a complete general-purpose sign-language translator.

## What it demonstrates

- Webcam-based frame processing with OpenCV.
- Background / foreground segmentation experiments.
- Hand-gesture classification using TensorFlow-based ML.
- Conversion of recognized gestures into text and speech.
- Experimental voice-to-sign interaction.

## Processing Flow

```text
Webcam
  ↓
Frame Capture
  ↓
Pre-processing
  ↓
Gesture Detection
  ↓
ML Classification
  ↓
Text
  ↓
Optional Text-to-Speech
```

## Current Scope

The model recognizes a **limited set of predefined gestures**. Recognition quality therefore depends on the trained classes, lighting, camera position, and background conditions.

This repository should be considered a learning/research prototype rather than a production-ready translation system.

## Limitations

- Limited gesture vocabulary.
- Recognition can degrade with lighting and background changes.
- Classification latency depends on hardware and model configuration.
- Full natural-language sign-language translation is not implemented.

## Getting Started

```bash
git clone https://github.com/Ajayprakashk7/sign-language-translation.git
cd sign-language-translation

pip install -r requirements.txt
```

Run the available prototype entry points:

```bash
python mic.py
python final.py
```

## Results

Example outputs are available in the repository's `assets/` directory.

## Future Work

- Expand the gesture dataset.
- Replace simple gesture classes with sequence-based recognition.
- Improve robustness across users and environments.
- Add quantitative evaluation metrics.
- Explore MediaPipe / landmark-based pipelines.
- Separate recognition from language-level translation.

## License

MIT

## Author

**Ajay Prakash** · [Portfolio](https://ajayprakash.dev) · [GitHub](https://github.com/Ajayprakashk7)
