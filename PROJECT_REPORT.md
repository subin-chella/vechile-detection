# Project Report: Automatic License Plate Recognition System

---

## **INDEX**

| Chapter No. | Title                  | Page No. |
|-------------|------------------------|----------|
| 1           | INTRODUCTION           | 5        |
| 1.1         | Problem Statement      | 7        |
| 1.2         | Project Overview       | 10       |
| 1.3         | Abstract               | 11       |
| 2           | SYSTEM ANALYSIS        | 12       |
| 2.1         | Feasibility Study      | 12       |
| 2.2         | Existing System        | 15       |
| 2.3         | Proposed System        | 17       |
| 3           | SYSTEM CONFIGURATION   | 20       |
| 3.1         | Hardware Specification | 21       |
| 3.2         | Software Specification | 22       |
| 3.3         | About the Software     | 24       |
| 4           | SYSTEM DESIGN          | 40       |
| 4.1         | System Architecture    | 41       |
| 4.2         | Data Flow Diagram (DFD)| 43       |
| 4.3         | Database Design        | 45       |
| 4.4         | Input Design           | 48       |
| 4.5         | Output Design          | 49       |
| 5           | SYSTEM DESCRIPTION     | 50       |
| 5.1         | Module Description     | 51       |
| 6           | TESTING AND IMPLEMENTATION | 70       |
| 6.1         | Testing Strategies     | 71       |
| 6.2         | Implementation         | 75       |
| 7           | CONCLUSION AND FUTURE SCOPE | 78       |
| 8           | BIBLIOGRAPHY           | 80       |

---

## **CHAPTER 1: INTRODUCTION**

Automatic License Plate Recognition (ALPR) is a technology that uses optical character recognition (OCR) on images to read vehicle registration plates to create vehicle location data. It can use existing closed-circuit television, road-rule enforcement cameras, or cameras specifically designed for the task. ALPR is used by police forces for law enforcement purposes, including to check if a vehicle is registered or licensed. It is also used for electronic toll collection on pay-per-use roads and as a method of cataloging the movements of traffic, for example, by highways agencies.

The ALPR process can be divided into several stages:
1.  **Vehicle Image Capture:** This is the first step where a camera captures the image of a vehicle. The quality of the image is crucial for the accuracy of the recognition process.
2.  **License Plate Detection:** In this stage, the captured image is processed to find the location of the license plate. This is a challenging task due to the variations in plate size, location, and the presence of other objects in the image.
3.  **Character Segmentation:** Once the license plate is detected, the individual characters on the plate are segmented for recognition.
4.  **Character Recognition:** In this final stage, each segmented character is recognized using an OCR engine.

This project implements a complete ALPR system that takes a video file as input, detects and tracks vehicles, and recognizes their license plates. The system is built using state-of-the-art deep learning models and computer vision libraries, and it provides a user-friendly web interface for easy interaction.

### **1.1 Problem Statement**

The manual process of identifying and recording license plate information is time-consuming, prone to errors, and inefficient, especially in high-traffic environments. An automated system is needed to accurately and efficiently detect and recognize license plates from video streams in real-time. This system should be able to handle various challenges, such as different lighting conditions, vehicle speeds, and plate designs.

### **1.2 Project Overview**

This project aims to develop a robust and accurate Automatic License Plate Recognition (ALPR) system. The system is designed to process video files, identify vehicles, track their movements, and recognize their license plates. The core of the system is a pipeline of deep learning models and computer vision algorithms that work together to achieve this goal. The project also includes a web-based user interface built with Streamlit, which allows users to easily upload videos, initiate the recognition process, and view the results.

The main objectives of the project are:
- To build a system that can accurately detect and track vehicles in a video.
- To implement a module for detecting and recognizing license plates with high accuracy.
- To provide a user-friendly interface for interacting with the system.
- To generate an output video with the recognized license plates and vehicle tracks overlaid on the original video.

### **1.3 Abstract**

This project presents an end-to-end Automatic License Plate Recognition (ALPR) system that processes video files to detect, track, and recognize vehicle license plates. The system leverages a combination of state-of-the-art deep learning models, including YOLOv8 for object detection and a custom-trained model for license plate detection, along with the DeepSORT algorithm for real-time vehicle tracking. The recognized characters are extracted using the EasyOCR library. The entire pipeline is orchestrated by a Python-based application with a user-friendly web interface created using Streamlit. The system is designed to be modular and extensible, allowing for future improvements and adaptations. The final output is an annotated video file showing the tracked vehicles and their recognized license plates.

---

## **CHAPTER 2: SYSTEM ANALYSIS**

### **2.1 Feasibility Study**

A feasibility study was conducted to assess the viability of the project. The study focused on three main areas: technical feasibility, economic feasibility, and operational feasibility.

#### **2.1.1 Technical Feasibility**

The technical feasibility study concluded that the project is technically feasible. The required technologies, including Python, OpenCV, YOLOv8, DeepSORT, and EasyOCR, are all mature and well-documented. The availability of pre-trained models for object detection and the ease of use of libraries like Streamlit make the development process straightforward. The main technical challenges identified were related to the accuracy of the OCR process, especially in handling low-quality images and non-standard license plates. These challenges were addressed by implementing a pipeline of image preprocessing techniques and a confidence-based locking mechanism.

#### **2.1.2 Economic Feasibility**

The project is economically feasible as it primarily relies on open-source software and libraries. The main cost associated with the project is the hardware required for development and deployment. However, the system can be run on a standard personal computer, and for larger-scale deployments, cloud-based solutions can be used to manage costs effectively. The potential benefits of the system, such as automating the process of license plate monitoring, outweigh the development and deployment costs.

#### **2.1.3 Operational Feasibility**

The proposed system is operationally feasible. The Streamlit-based web interface is designed to be intuitive and easy to use, requiring minimal training for the end-users. The system automates the entire process of license plate recognition, from video upload to the generation of the final report, thus reducing the manual effort required.

### **2.2 Existing System**

The "existing system" can be defined as the manual process of monitoring and recording license plate information. This process typically involves a human operator who manually watches video feeds and records the license plate numbers.

The main drawbacks of the existing system are:
- **Time-consuming:** The manual process is very slow and not suitable for high-traffic environments.
- **Error-prone:** Human operators are prone to making errors, especially when dealing with a large volume of data.
- **Inefficient:** The manual process is not scalable and requires a significant amount of human resources.

### **2.3 Proposed System**

The proposed system is a fully automated ALPR system that overcomes the limitations of the manual process. The system is designed to be accurate, efficient, and easy to use.

The main features of the proposed system are:
- **Automated Detection and Recognition:** The system automatically detects vehicles and recognizes their license plates from video files.
- **High Accuracy:** The use of state-of-the-art deep learning models ensures high accuracy in both detection and recognition.
- **Real-time Processing:** The system is designed to process video streams in near real-time.
- **User-friendly Interface:** The Streamlit-based web interface makes it easy for users to interact with the system.
- **Scalability:** The modular architecture of the system allows for easy scalability and future enhancements.

---

## **CHAPTER 3: SYSTEM CONFIGURATION**

### **3.1 Hardware Specification**

The following are the recommended hardware specifications for running the ALPR system:

- **Processor:** Intel Core i5 or equivalent (Intel Core i7 or equivalent recommended for better performance).
- **RAM:** 8 GB (16 GB or more recommended).
- **Storage:** 500 GB of free disk space for storing video files and the application data.
- **GPU:** An NVIDIA GPU with CUDA support is highly recommended for real-time processing, although the system can also run on a CPU.

### **3.2 Software Specification**

The ALPR system is developed using Python and relies on several open-source libraries. The following are the main software requirements:

- **Operating System:** Windows, macOS, or Linux.
- **Python:** Version 3.10 or higher.
- **Libraries:** The required Python libraries are listed in the `requirements.txt` file. The main libraries are:
    - `streamlit`
    - `ultralytics`
    - `deep-sort-realtime`
    - `easyocr`
    - `opencv-python`
    - `numpy`
    - `pandas`

### **3.3 About the Software**

This section provides a brief overview of the key software libraries used in the project.

#### **3.3.1 Streamlit**

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In this project, Streamlit is used to build the user interface of the ALPR system. The UI allows users to upload video files, start the processing, and download the results.

#### **3.3.2 YOLOv8 (Ultralytics)**

YOLOv8 is the latest version of the You Only Look Once (YOLO) family of real-time object detection models. It is known for its high accuracy and speed. In this project, YOLOv8 is used for two purposes:
1.  **Vehicle Detection:** A pre-trained YOLOv8 model is used to detect vehicles (cars, trucks, etc.) in the video frames.
2.  **License Plate Detection:** A custom-trained YOLOv8 model is used to detect license plates on the detected vehicles.

#### **3.3.3 DeepSORT**

DeepSORT is a popular tracking algorithm that is used for multi-object tracking. It is an extension of the SORT (Simple Online and Realtime Tracking) algorithm that adds a deep learning component to improve tracking performance, especially in cases of occlusion. In this project, DeepSORT is used to track the detected vehicles across the video frames, assigning a unique ID to each vehicle.

#### **3.3.4 EasyOCR**

EasyOCR is a Python library for Optical Character Recognition (OCR). It is designed to be easy to use and supports a wide range of languages. In this project, EasyOCR is used to recognize the characters on the detected license plates.

#### **3.3.5 OpenCV**

OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision. In this project, OpenCV is used for various image and video processing tasks, such as reading and writing video files, converting color spaces, and drawing bounding boxes.

#### **3.3.6 NumPy and Pandas**

NumPy and Pandas are two fundamental libraries for scientific computing and data analysis in Python.
- **NumPy** is used for numerical operations, especially for handling the multi-dimensional arrays that represent images and bounding boxes.
- **Pandas** is used for data manipulation and analysis. Although not heavily used in the core processing pipeline, it can be useful for analyzing the output data.

---

## **CHAPTER 4: SYSTEM DESIGN**

### **4.1 System Architecture**

The ALPR system is designed with a modular architecture, where each component has a specific responsibility. This modular design makes the system easy to understand, maintain, and extend. The main components of the system are:

1.  **User Interface (`app.py`):** This is the front-end of the application, built with Streamlit. It provides a simple and intuitive interface for users to interact with the system.
2.  **Video Processing Engine (`processing.py`):** This is the core of the system, responsible for orchestrating the entire video processing pipeline.
3.  **Object Detection and Tracking Module (`processing.py`):** This module uses YOLOv8 and DeepSORT to detect and track vehicles.
4.  **License Plate Recognition Module (`util.py`):** This module uses a custom YOLOv8 model and EasyOCR to detect and recognize license plates.
5.  **Data Interpolation Module (`interpolation.py`):** This module is used to fill in the gaps in the tracking data, resulting in smoother and more consistent tracking.
6.  **Visualization Module (`visualization.py`):** This module generates the final output video with the detected bounding boxes and license plate numbers.
7.  **Database Module (`database.py`):** This module manages the in-memory SQLite database that is used to store the detection and tracking data.

### **4.2 Data Flow Diagram (DFD)**

The following is a textual representation of the Data Flow Diagram (DFD) for the ALPR system:

```
Level 0:
+-----------------+     video_file     +----------------------+
|      User       |------------------->| ALPR System (app.py) |
+-----------------+                    +----------------------+
                                                 |
                                                 | output_video
                                                 v
                                           +----------+
                                           | User     |
                                           +----------+

Level 1:
+-----------------+     video_file     +----------------------+
|      User       |------------------->|   Streamlit UI       |
+-----------------+                    |      (app.py)        |
                                       +----------------------+
                                                 |
                                                 | video_path
                                                 v
                                       +----------------------+
                                       | Video Processing     |
                                       |   Engine (processing.py) |
                                       +----------------------+
                                                 |
                                                 | detection_data
                                                 v
                                       +----------------------+
                                       | Data Interpolation   |
                                       | (interpolation.py)   |
                                       +----------------------+
                                                 |
                                                 | interpolated_data
                                                 v
                                       +----------------------+
                                       | Visualization        |
                                       |   (visualization.py) |
                                       +----------------------+
                                                 |
                                                 | output_video
                                                 v
                                       +----------------------+
                                       |   Streamlit UI       |
                                       |      (app.py)        |
                                       +----------------------+
                                                 |
                                                 | output_video
                                                 v
                                           +----------+
                                           | User     |
                                           +----------+
```

### **4.3 Database Design**

The system uses an in-memory SQLite database to store the detection and tracking data for each video processing session. This approach is chosen for its simplicity and performance, as it avoids the overhead of writing to and reading from disk during the processing pipeline.

The database consists of three main tables:

1.  **`videos`:** This table stores information about the processed videos.
    - `id` (TEXT, PRIMARY KEY): A unique identifier for each video.
    - `filename` (TEXT): The original filename of the uploaded video.
    - `output_path` (TEXT): The path to the generated output video.
    - `created_at` (TIMESTAMP): The timestamp when the video was registered.

2.  **`detections`:** This table stores the raw detection data for each frame.
    - `video_id` (TEXT): A foreign key referencing the `videos` table.
    - `frame_nmr` (INTEGER): The frame number.
    - `car_id` (INTEGER): The unique ID of the tracked vehicle.
    - `car_bbox` (TEXT): The bounding box of the vehicle (stored as a JSON string).
    - `license_plate_bbox` (TEXT): The bounding box of the license plate (stored as a JSON string).
    - `license_plate_bbox_score` (REAL): The confidence score of the license plate detection.
    - `license_number` (TEXT): The recognized license plate number.
    - `license_number_score` (REAL): The confidence score of the OCR.

3.  **`interpolated_detections`:** This table stores the interpolated detection data. It has the same schema as the `detections` table, with an additional column:
    - `is_imputed` (INTEGER): A flag indicating whether the detection was interpolated (1) or original (0).

### **4.4 Input Design**

The primary input to the system is a video file. The user can upload a video file in one of the supported formats (MP4, AVI, MOV) through the Streamlit web interface. The application is designed to handle videos of various resolutions and frame rates.

### **4.5 Output Design**

The main output of the system is an annotated video file in MP4 format. This video shows the original video with the following information overlaid:
- Bounding boxes around the tracked vehicles.
- Bounding boxes around the detected license plates.
- The recognized license plate number displayed above the vehicle.

In addition to the video output, the system also generates a CSV file (`test.csv`) containing the detailed detection and tracking data for each frame. This file can be used for debugging and analysis purposes.

---

## **CHAPTER 5: SYSTEM DESCRIPTION**

This chapter provides a detailed description of the system's implementation, focusing on the key modules and their functionalities.

### **5.1 Module Description**

#### **5.1.1 `app.py` - The Streamlit User Interface**

This is the main entry point of the application. It uses the Streamlit library to create a web-based user interface. The main functionalities of this module are:
- **File Upload:** It provides a file uploader widget that allows users to upload video files.
- **Processing Orchestration:** When the user clicks the "Process Video" button, this module orchestrates the entire video processing pipeline by calling the functions from the other modules in the correct order:
    1.  `register_video()`: Registers the uploaded video in the database.
    2.  `process_video()`: Starts the core video processing.
    3.  `interpolate_results()`: Interpolates the detection results.
    4.  `visualize_results()`: Generates the output video.
- **Progress and Status Display:** It displays a progress bar and status messages to keep the user informed about the progress of the video processing.
- **Result Display and Download:** Once the processing is complete, it displays the output video and provides a download button for the user to save the video.

#### **5.1.2 `processing.py` - The Core Processing Engine**

This module contains the core logic for video processing. The `process_video` function is the main function in this module. It performs the following steps:
1.  **Initialization:** It initializes the YOLOv8 models for vehicle and license plate detection, and the DeepSORT tracker.
2.  **Video Reading:** It reads the input video frame by frame.
3.  **Vehicle Detection and Tracking:** For each frame, it performs the following:
    - It uses the `coco_model` (YOLOv8) to detect vehicles.
    - It passes the detected vehicle bounding boxes to the `mot_tracker` (DeepSORT) to get the tracked vehicles with unique IDs.
4.  **License Plate Detection and Recognition:** For each tracked vehicle, it performs the following:
    - It uses the `license_plate_detector` (custom YOLOv8) to detect the license plate.
    - It crops the license plate region from the frame.
    - It performs image preprocessing on the cropped license plate (grayscale conversion, bilateral filter, Otsu's thresholding, resizing).
    - It calls the `read_license_plate` function from `util.py` to perform OCR on the preprocessed image.
5.  **Result Storage:** It stores the detection and tracking results (frame number, car ID, bounding boxes, license plate text, and scores) in the SQLite database by calling the `store_detections` function.

#### **5.1.3 `util.py` - Utility Functions and OCR**

This module provides a collection of utility functions used throughout the project. The most important function in this module is `read_license_plate`, which is responsible for the OCR part of the pipeline.
The `read_license_plate` function:
1.  Takes a preprocessed image of a license plate as input.
2.  Uses the `easyocr.Reader` to read the text from the image. It uses a character `allowlist` to restrict the OCR to alphanumeric characters.
3.  It then generates several candidates for the license plate number by applying character substitution rules (e.g., 'O' to '0', 'I' to '1').
4.  It validates the candidates against a set of predefined patterns for Indian license plates.
5.  It returns the best candidate with the highest confidence score.

This module also contains functions for validating and formatting license plate numbers based on Indian standards.

#### **5.1.4 `interpolation.py` - Data Interpolation**

This module is responsible for improving the smoothness of the tracking by interpolating the bounding box data for the frames where a vehicle was not detected. The `interpolate_results` function:
1.  Fetches the raw detection data from the database.
2.  For each tracked vehicle, it identifies the frames where the detection is missing.
3.  It uses linear interpolation (`scipy.interpolate.interp1d`) to estimate the bounding box coordinates for the missing frames.
4.  It stores the interpolated data back into the `interpolated_detections` table in the database.

#### **5.1.5 `visualization.py` - Output Video Generation**

This module is responsible for creating the final output video. The `visualize_results` function:
1.  Fetches the (interpolated) detection data from the database.
2.  Reads the original input video frame by frame.
3.  For each frame, it draws the bounding boxes for the tracked vehicles and their license plates.
4.  It also displays the recognized license plate number above each vehicle.
5.  It writes the annotated frames to a new video file.

#### **5.1.6 `database.py` - Database Management**

This module (inferred from the code) is responsible for all the interactions with the SQLite database. It provides functions for:
- Creating a connection to the database.
- Initializing the database schema (creating the tables).
- Storing and fetching detection and interpolation data.
- Registering and managing video information.

---

## **CHAPTER 6: TESTING AND IMPLEMENTATION**

### **6.1 Testing Strategies**

The ALPR system was tested using a combination of unit testing, integration testing, and end-to-end testing.

#### **6.1.1 Unit Testing**

Unit tests were written for the individual functions in the `util.py` and `interpolation.py` modules. For example:
- The `license_complies_format` function in `util.py` was tested with a variety of valid and invalid license plate numbers to ensure that it correctly validates the format.
- The interpolation logic in `interpolation.py` was tested with sample data to verify that it correctly interpolates the missing bounding boxes.

#### **6.1.2 Integration Testing**

Integration tests were performed to ensure that the different modules of the system work together correctly. For example:
- The integration between the `processing.py` module and the `util.py` module was tested to ensure that the preprocessed license plate images are correctly passed to the OCR function.
- The integration between the `processing.py`, `interpolation.py`, and `visualization.py` modules was tested to ensure that the data flows correctly from detection to interpolation to visualization.

#### **6.1.3 End-to-End Testing**

End-to-end testing was performed by running the complete application with a set of test videos. The output videos and the generated CSV files were manually inspected to verify the accuracy of the vehicle detection, tracking, and license plate recognition. The user interface was also tested to ensure that the file upload, processing, and download functionalities work as expected.

### **6.2 Implementation**

The ALPR system is implemented as a Python application with a Streamlit web interface. The following are the steps to set up and run the application:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
4.  **Access the application:** Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

Once the application is running, the user can upload a video file, click the "Process Video" button, and wait for the processing to complete. The annotated video can then be downloaded from the web interface.

---

## **CHAPTER 7: CONCLUSION AND FUTURE SCOPE**

This project successfully demonstrates the development of an end-to-end Automatic License Plate Recognition system. The system is capable of processing video files, detecting and tracking vehicles, and recognizing their license plates. The use of a modular architecture and state-of-the-art deep learning models makes the system robust and accurate.

However, there is always room for improvement. The main limitation of the current system is the accuracy of the OCR, which is highly dependent on the quality of the input video and the performance of the `easyocr` library.

Future work could focus on the following areas:
- **Improving OCR Accuracy:** This could be achieved by:
    - Training a custom OCR model on a large dataset of license plate images.
    - Implementing more advanced image enhancement techniques, such as super-resolution and deblurring.
    - Experimenting with different OCR engines.
- **Real-time Performance:** The system could be optimized for real-time performance by using more lightweight models or by deploying it on more powerful hardware.
- **Support for More Plate Formats:** The system could be extended to support license plate formats from different countries and regions.
- **Deployment:** The application could be deployed as a cloud-based service to make it accessible to a wider audience.

---

## **CHAPTER 8: BIBLIOGRAPHY**

- **YOLOv8:** Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.
- **DeepSORT:** Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. In 2017 IEEE international conference on image processing (ICIP) (pp. 3645-3649). IEEE.
- **EasyOCR:** Jaided AI, EasyOCR. https://github.com/JaidedAI/EasyOCR
- **Streamlit:** Streamlit Inc., Streamlit Documentation. https://docs.streamlit.io/
- **OpenCV:** Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.