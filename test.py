# Import necessary libraries
# import cvzone # cvzone for additional OpenCV utilities
from ultralytics import YOLO  # Importing YOLO class for object detection
import cv2 as cv  # OpenCV for image processing

# Load the YOLO model
model = YOLO('runs/classify/train9/weights/best.pt')  # Initialize YOLO model with pre-trained weights

# Define class names for detected objects
# classNames = ['George', 'Unlabeled']  # Class names corresponding to the model's training

# Read the image
image_path = "D:/my learning/Object detection using yolo V8 and roboflow/George-detection.v1i.folder/test/Unlabeled/452308fdef2b11b747eb712f09507d32_jpg.rf.501e8a0806806844913e51a0cc01d964.jpg"  # Path to the image file
img = cv.imread(image_path)  # Read the image using OpenCV

# Perform object detection on the image
results = model(image_path)  # Object detection with YOLO model

# Display the processed image
cv.imshow("Image", img)
cv.waitKey(0)  # Wait for a key press to close the image window

