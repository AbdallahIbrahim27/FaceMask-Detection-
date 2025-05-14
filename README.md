# Face Mask Detection

This project detects whether people in a video stream are wearing face masks using deep learning and computer vision techniques. It uses a pre-trained MobileNetV2 model for mask detection and OpenCV for real-time video processing.

## Overview
Face Mask Detection is a computer vision application that automatically detects faces in a video stream and classifies each detected face as either "Mask" or "No Mask". This is useful for monitoring mask compliance in public spaces, workplaces, or any environment where mask-wearing is required.

The system consists of two main components:
1. **Face Detection**: Uses a pre-trained Caffe model (Single Shot Multibox Detector - SSD) to locate faces in each video frame.
2. **Mask Classification**: Uses a deep learning model (MobileNetV2) trained to distinguish between faces with and without masks.

## How It Works
- The script captures video from your webcam in real time.
- Each frame is processed to detect faces using OpenCV's DNN module and the SSD face detector.
- For each detected face, the region of interest is extracted, preprocessed, and passed to the mask detection model.
- The model predicts whether the person is wearing a mask or not.
- The result is displayed on the video stream with bounding boxes and labels.

## Model Details
- **Face Detector**: The face detector is based on the SSD framework with a ResNet-10 backbone, provided by OpenCV. The model files are:
  - `face_detector/deploy.prototxt`: Defines the model architecture.
  - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained weights.
- **Mask Detector**: The mask detector is a Keras model based on MobileNetV2, a lightweight and efficient convolutional neural network. It is trained on a dataset of face images with and without masks.

## Features
- Real-time face mask detection from webcam video
- Uses a deep learning model trained on images of people with and without masks
- Displays bounding boxes and labels for detected faces

## Requirements
- Python 3.7+
- TensorFlow >= 2.4.0
- OpenCV >= 4.5.0
- imutils >= 0.5.4
- numpy >= 1.19.5

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Files
- `detect_mask_video.py`: Main script to run real-time mask detection from your webcam.
- `mask_detector.h5`: Pre-trained Keras model for mask detection.
- `face_detector/`: Contains the face detection model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
- `train_mask_detector.py`: Script used to train the mask detection model (optional, for retraining).

## Usage
1. Make sure your webcam is connected.
2. Open a terminal in the project directory.
3. Run the following command:
   ```bash
   python detect_mask_video.py
   ```
4. A window will open showing the webcam feed with detected faces and mask/no mask labels.
5. Press `q` to quit the application.

## Notes
- The model file must be named `mask_detector.h5` and be in the same directory as `detect_mask_video.py`.
- The face detector files must be in the `face_detector` directory.

## Training the Model (Optional)
If you want to retrain the mask detection model:
1. Prepare a dataset with two folders: `with_mask` and `without_mask`, each containing images of faces.
2. Update the `DIRECTORY` variable in `train_mask_detector.py` to point to your dataset location.
3. Run the training script:
   ```bash
   python train_mask_detector.py
   ```
4. The script will train a new model and save it as `mask_detector.h5`.

## Applications
- Public health monitoring
- Workplace safety
- Automated compliance systems
- Educational projects in computer vision and deep learning

## License
This project is for educational and research purposes. You are free to use, modify, and distribute it for non-commercial applications. For commercial use, please check the licenses of the underlying models and datasets. 
