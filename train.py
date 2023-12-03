# Import necessary libraries and modules
from torch.fx.experimental.unification import variables  # Importing 'variables' from PyTorch FX library for transformations
from ultralytics import YOLO  # Importing YOLO class for object detection from Ultralytics
import torch  # Importing PyTorch, a deep learning framework

# Clearing CUDA memory cache
torch.cuda.empty_cache()  # Clears unused memory from GPU to free up space and potentially enhance performance

# Importing garbage collection module for manual memory management
import gc  # Garbage Collection to manage memory manually

# Memory Management
del variables  # Deleting 'variables' to free up memory, useful if 'variables' are no longer needed
gc.collect()  # Manually triggering garbage collection to free up memory

# Displaying CUDA memory summary
torch.cuda.memory_summary(device=None, abbreviated=False)  # Provides a detailed report on CUDA memory usage

# Load and use a YOLO model
if __name__ == '__main__':  # Ensures the following code runs only if the script is the main program
    model = YOLO(model="yolov8n-cls.pt", task='classify')  # Initializes a YOLO model using configuration from 'yolov8n.yaml'

    # Training the model
    results = model.train(data="D:/my learning/Object detection using yolo V8 and roboflow/George-detection.v1i.folder", epochs=14, workers = 2)  # Trains the model on data specified in 'data.yaml' for 1 epoch




# The YOLOv8 series offers different models for object detection. 
# YOLOv8n is the fastest and lightest with lower accuracy (mAP 37.3) and minimal computational needs. yolov8n.yaml to be run
# YOLOv8m balances performance and resources, offering better accuracy (mAP 50.2) but requires more computational power. yolov8m.yaml
# YOLOv8l, the most resource-intensive, provides the highest accuracy (mAP 52.9) but demands significantly more computational resources. yolov8l.yaml
# Each model is suited for different scenarios, from fast and light applications to those needing high accuracy at the cost of computational efficiency.