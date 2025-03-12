import os
import cv2
import numpy as np
import h5py
from skimage.feature import graycomatrix, graycoprops

# Folder names for joints and cracks
joint_folder = "Joints"
crack_folder = "Cracks"

# Target image size for resizing
img_width, img_height = 640, 480

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    for file in filenames:
        img_path = os.path.join(folder,file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
    return images

# Function to compute Color Histogram (16 bins per channel → 48 values)
def colour_histogram(image, bins = 16):
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        hist_features.extend(hist)
    return hist_features

# Function to compute the Hu Moments (7 values)
def hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return hu

# Function to compute Haralick Texture Features (GLCM)
def haralick_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                      levels = 256, symmetric = True, normed = True)
    
    # Texture features for each angle so 4 values for each feature
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10), axis = (0,1)).flatten()

    return np.hstack([contrast, correlation, energy, homogeneity, entropy])  # 20 values

# Function to extract all features from an image
def extract_features(image):
    colour_hist = colour_histogram(image)
    hu_mom = hu_moments(image)
    haralick = haralick_features(image)
    return np.hstack([colour_hist, hu_mom, haralick])  # 75 values (48 + 7 + 20)

# Loading images from the two folders
joint_images = load_images_from_folder(joint_folder)
crack_images = load_images_from_folder(crack_folder)

# Extracting features for all joint and crack images
joint_features = np.array([extract_features(img) for img in joint_images])
crack_features = np.array([extract_features(img) for img in crack_images])

# Labels: 0 for joints, 1 for cracks
joint_labels = np.zeros(joint_features.shape[0])
crack_labels = np.ones(crack_features.shape[0])

X = np.vstack([joint_features, crack_features])
y = np.hstack([joint_labels, crack_labels])

# Saving the features and labels to an HDF5 file
with h5py.File("dataset.h5", "w") as out:
    out.create_dataset("X", data = X)
    out.create_dataset("y", data = y)

print(f"Feature Matrix Shape: {X.shape}, Label Vector Shape: {y.shape}")
print(f"Data saved successfully to {os.getcwd()}")