# Technical Documentation: Automatic License Plate Recognition using YOLOv8

---

### **1. Introduction**

**1.1. Project Overview**

This document provides a detailed technical overview of the Automatic License Plate Recognition (ALPR) system. The system is designed to process video files, detect and track vehicles, and recognize their license plates in real-time. The application is built using Python and features a user-friendly web interface created with Streamlit.

**1.2. Purpose**

The primary purpose of this project is to create a robust and accurate ALPR system by leveraging state-of-the-art deep learning models and computer vision techniques. The system is designed to be a practical demonstration of how these technologies can be applied to solve real-world problems.

**1.3. Scope**

The scope of this project includes:
*   The development of a web application for video upload and processing.
*   The integration of object detection and tracking algorithms for vehicles.
*   The implementation of a license plate detection and recognition module.
*   The generation of an output video with the processed results.

---

### **2. System Architecture**

**2.1. High-Level Architecture**

The system follows a modular architecture, with each component responsible for a specific task. The high-level architecture can be broken down as follows:

```
[Video Input] -> [User Interface (Streamlit)] -> [Video Processing Engine] -> [Object Detection & Tracking] -> [License Plate Recognition] -> [Output Video]
```

**2.2. Component-wise Breakdown**

*   **2.2.1. User Interface (Streamlit):**
    *   The front-end of the application is a web interface built with Streamlit. It allows users to upload video files, initiate the processing, and view the results.
*   **2.2.2. Video Processing Engine:**
    *   This is the core component that orchestrates the entire video processing pipeline. It reads the video frame by frame and passes each frame to the object detection and tracking module.
*   **2.2.3. Object Detection and Tracking Module:**
    *   This module is responsible for detecting and tracking vehicles in the video. It uses the YOLOv8 model for vehicle detection and the DeepSORT algorithm for tracking.
*   **2.2.4. License Plate Recognition Module:**
    *   This module is responsible for detecting and recognizing the license plates of the tracked vehicles. It uses a custom-trained YOLOv8 model for license plate detection and the EasyOCR library for character recognition.

---

### **3. Core Modules and Algorithms**

**3.1. Vehicle Detection**

*   **3.1.1. Model: YOLOv8**
    *   We use the YOLOv8 model, pre-trained on the COCO dataset, for vehicle detection. YOLOv8 is a state-of-the-art, real-time object detection model known for its high accuracy and speed.
*   **3.1.2. Implementation Details:**
    *   The `ultralytics` Python library is used to load and run the YOLOv8 model.
    *   The model is configured to detect a predefined set of vehicle classes (e.g., car, truck, bus).

**3.2. Vehicle Tracking**

*   **3.2.1. Algorithm: DeepSORT**
    *   DeepSORT is used for real-time multi-object tracking. It is an extension of the SORT (Simple Online and Realtime Tracking) algorithm that integrates a deep learning model to improve tracking accuracy, especially in scenarios with occlusions.
*   **3.2.2. Implementation Details:**
    *   The `deep-sort-realtime` Python library is used for the DeepSORT implementation.
    *   The tracker is initialized with a `max_age` parameter, which defines the maximum number of frames to keep tracking a lost object before it is considered gone.

**3.3. License Plate Detection**

*   **3.3.1. Model: Custom-trained YOLOv8**
    *   A custom YOLOv8 model is used for license plate detection. This model has been trained on a specific dataset of images with annotated license plates to achieve high accuracy in locating license plates on vehicles.
*   **3.3.2. Implementation Details:**
    *   The custom-trained model is loaded and used in the same way as the vehicle detection model.
    *   The license plate detector is run on the region of interest (the bounding box of the tracked vehicle) to improve efficiency.

**3.4. License Plate Recognition (OCR)**

*   **3.4.1. Library: EasyOCR**
    *   The EasyOCR library is used for optical character recognition. It is a powerful and user-friendly library that provides high accuracy in reading text from images.
*   **3.4.2. Implementation Details:**
    *   The license plate region is cropped from the frame and preprocessed (converted to grayscale and thresholded) to improve OCR accuracy.
    *   EasyOCR is then used to read the characters from the preprocessed image.
*   **3.4.3. Post-processing and Filtering:**
    *   To improve the stability of the recognized license plate numbers, we implement a confidence-based filtering mechanism.
    *   For each tracked vehicle, we store the license plate number with the highest confidence score over a series of frames. This "best" license plate number is then displayed, which prevents the text from flickering or changing in every frame.

---

### **4. Codebase Structure**


**4.1. Key Scripts and their Functions**

*   **`app.py`:** The main entry point of the application. It contains the Streamlit code for the user interface and orchestrates the overall workflow.
*   **`processing.py`:** This script contains the core video processing logic, including vehicle detection, tracking, and license plate recognition.
*   **`visualization.py`:** This script is responsible for generating the output video. It draws the bounding boxes for the vehicles and license plates and displays the recognized license plate numbers.
*   **`util.py`:** A collection of utility functions used throughout the project, such as functions for reading and writing CSV files and formatting the license plate text.
*   **`config.py`:** This file contains the configuration parameters for the application, such as the number of seconds of the video to process.
*   **`requirements.txt`:** A list of the Python libraries and dependencies required to run the project.

---

### **5. Data Flow**

**5.1. Video Upload and Pre-processing**

1.  The user uploads a video file through the Streamlit web interface.
2.  The video is saved to a temporary location on the server.

**5.2. Frame-by-Frame Processing**

1.  The `processing.py` script reads the video frame by frame.
2.  For each frame:
    *   The YOLOv8 model detects vehicles.
    *   The DeepSORT algorithm updates the tracks of the vehicles.
    *   For each tracked vehicle, the custom YOLOv8 model detects the license plate.
    *   The license plate is cropped and preprocessed.
    *   EasyOCR reads the characters from the license plate.
    *   The results (bounding boxes, track IDs, license plate numbers, and confidence scores) are stored in a dictionary.

**5.3. Results Aggregation and Output**

1.  The results from the processing step are saved to a CSV file.
2.  The `visualization.py` script reads the CSV file and the original video.
3.  It then generates a new video with the bounding boxes and license plate numbers drawn on each frame.
4.  The final video is made available for download through the Streamlit interface.

---

### **6. Setup and Deployment**

**6.1. Prerequisites**

*   Python 3.10 or higher
*   pip

**6.2. Installation**

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Install the required dependencies: `pip install -r requirements.txt`

**6.3. Running the Application**

1.  Run the Streamlit application: `streamlit run app.py`
2.  Open the application in a web browser at the provided URL.

---

### **7. Future Work and Scalability**

**7.1. Potential Improvements**

*   **Model Optimization:** The YOLOv8 and DeepSORT models can be further optimized for better performance on low-end hardware.
*   **Dataset Expansion:** The custom license plate detection model can be trained on a more diverse dataset to improve its accuracy and robustness.
*   **Internationalization:** The system can be extended to support license plate formats from different countries and regions.

**7.2. Scalability Considerations**

*   **Cloud Deployment:** For a production environment, the application can be deployed to a cloud server (e.g., AWS, Google Cloud) to handle a larger number of users and videos.
*   **Distributed Processing:** For very large videos, the processing could be distributed across multiple machines to reduce the processing time.
