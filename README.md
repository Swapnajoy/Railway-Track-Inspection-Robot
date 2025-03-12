# Railway-Track-Inspection-Robot
This repository contains the code for a robotic system that was developed as part of a team project during my Bachelor's degree. My main contribution focused on the **Machine Learning-based Computer Vision** approach for **crack detection on railway tracks**, including code for **image loading**, **feature extraction**, and **training classification models**.

The project was a collaborative effort with a total of six contributors.The project was awarded a **patent** under the title "A ROBOTIC SYSTEM FOR INSPECTION OF RAILWAY TRACKS" (Patent granted December 2024 by The Patent Office, Government of India). You can find the patent certificate in the `docs/` folder.

## Overview
The work involves the design and development of railway tracks-inspection robotic system. The following hardware and software are used:

A. Hardware:

  i. For Inspection: A Raspberry pi (RPi) as controller, a Relay as motor driver and a 12V DC motor are used. A specially fabricated chassis made of steel is used as the inspecting vehicle with the above said components installed in it.
  
  ii. For Imaging system: Raspberry Pi (same module) as processor, a 5 MP webcam interfaced via USB port of RPi and a GPS module for geotagging and timestamping.
  
B. Software: 

Python and OpenCV are used for the ML and IP codes, respectively. 


## Project Structure

- `src/`: Contains the source code files.
  - `feature-extractor.py`: Code for loading images of joints and cracks, extracting the features and build the dataset with target labels.
  - `classification_models.py`: Code for testing different ML classification models (e.g., Logistic Regression, SVM).
  - `final_model.py`: Code for training the final model using **Logistic Regression** and saving it.
- `images/`: Contains the image of the assembled robot.
- `docs/`: Contains the patent certificate.
- `requirements.txt`: Lists all necessary Python libraries for the project.
- `.gitignore`: Lists files/folders to ignore.
