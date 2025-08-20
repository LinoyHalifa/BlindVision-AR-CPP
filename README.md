# BlindVision AR (C++ + OpenCV + YOLOv8 ONNX)

An assistive vision project designed to help visually impaired individuals:  
a camera detects real-world objects using YOLOv8 (ONNX format) and provides spoken feedback about their position.

---

## Features
- **OpenCV** integration for real-time camera capture and visualization with bounding boxes.  
- **YOLOv8 (ONNX)** for accurate object detection.  
- Object position analysis: *left / right / in front*.  
- Extendable with Text-to-Speech (TTS) for audible feedback.  

---

## Requirements
- Windows 10/11  
- Visual Studio 2022 (with CMake support)  
- OpenCV installed (set environment variable `OPENCV_DIR`, e.g., `C:\opencv\build`)  
- YOLOv8 ONNX model file (e.g., `yolov8n.onnx`)  

---

## Build Instructions
1. Clone or copy the `BlindVision AR_C++` project folder.  
2. Set the environment variable for OpenCV:  
   - `OPENCV_DIR=C:\opencv\build`  
3. Open the folder in **Visual Studio** → `File → Open → Folder…`.  
4. Select the configuration **x64-Release**.  
5. Build (`Ctrl+Shift+B`) → Run (`Ctrl+F5`).  

---

## Usage
- The camera window will open and display detected objects with bounding boxes.  
- The program outputs the position of one detected object per frame (*left / right / in front*).  
- Exit the application by pressing `Q` or `Esc`.  

---

## Project Structure

BlindVision AR_C++/
├── BlindVision.cpp # Main source code
├── CMakeLists.txt # CMake build configuration
├── yolov8n.onnx # YOLOv8 ONNX model
└── README.md # Project documentation

## Author
**Linoy Halifa**  