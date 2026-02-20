# 😄 Real-Time Facial Emotion Detection

A real-time facial emotion recognition system built with **OpenCV** and a **Keras/TensorFlow** deep learning model. The system detects faces via webcam using Haar Cascade classifiers and predicts one of seven core emotions with bounding box overlays.

---

## 🎯 Detected Emotions

| Label | Emotion |
|-------|---------|
| 😠 | Angry |
| 🤢 | Disgust |
| 😨 | Fear |
| 😄 | Happy |
| 😐 | Neutral |
| 😢 | Sad |
| 😲 | Surprise |

---

## 🗂️ Project Structure

```
emotion-detection/
│
├── emotion_detection.py              # Main script for real-time detection
├── emotion_model.hdf5                # Pre-trained CNN model (Keras/TF)
├── haarcascade_frontalface_default.xml  # OpenCV face detection classifier
└── README.md
```

---

## ⚙️ How It Works

1. **Face Detection** — OpenCV reads the webcam feed frame-by-frame and uses the Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to locate faces in real time.
2. **Preprocessing** — Detected face regions are converted to grayscale, resized to `48×48` pixels, and normalized to match the model's expected input.
3. **Emotion Prediction** — The preprocessed face is passed to the pre-trained CNN (`emotion_model.hdf5`) which outputs a probability distribution over 7 emotion classes.
4. **Overlay** — The predicted emotion label and confidence are drawn on the video frame as a bounding box annotation.

---

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) trained on the **FER-2013** dataset, which contains ~35,000 labeled grayscale facial images.

- **Input:** 48×48 grayscale image
- **Output:** Softmax over 7 emotion classes
- **Format:** Keras HDF5 (`.hdf5`)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```bash
pip install tensorflow keras opencv-python numpy
```

Or create a `requirements.txt`:

```
tensorflow>=2.0
keras
opencv-python
numpy
```

### Run

```bash
python emotion_detection.py
```

> Press **`q`** to quit the webcam window.

---

## 📋 Requirements

| Package | Purpose |
|--------|---------|
| `tensorflow` / `keras` | Load and run the `.hdf5` model |
| `opencv-python` | Webcam capture & face detection |
| `numpy` | Array manipulation & preprocessing |

---

## 💡 Use Cases

- Mental health & wellness monitoring tools
- Human-computer interaction (HCI) research
- Driver drowsiness / fatigue detection systems
- Customer experience analytics
- Interactive installations and art

---

## 📌 Notes

- Ensure your system has a working webcam before running.
- Lighting conditions significantly affect face detection accuracy.
- The Haar Cascade classifier works best with frontal, well-lit faces.
- For improved accuracy, consider replacing Haar Cascades with a DNN-based face detector (e.g., `dnn` module or MTCNN).

---

## 🔮 Future Improvements

- [ ] Replace Haar Cascade with a deep learning-based face detector (MTCNN / RetinaFace)
- [ ] Add support for video file input (not just webcam)
- [ ] Build a Flask/FastAPI web interface
- [ ] Retrain model on a larger, more diverse dataset
- [ ] Export model to TFLite for mobile deployment

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Kaggle
- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [Keras](https://keras.io/) / [TensorFlow](https://www.tensorflow.org/) — Deep Learning Framework
