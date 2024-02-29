from ultralytics import YOLO
import os
import torch
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("Training_Results/insta360_train1/weights/best.pt")  # load a pretrained model (recommended for training)
# Move the model to the GPU if available
model.to(device)
print(model.device)

for image_path in glob.glob(f'insta360_renamed_2024_01_31/verification_rename_dataset_2024-01-31-17-35-20/*.jpg'):
    print(image_path)
    result = model(source=image_path,conf=0.6)  # predict on an image
    print(result.boxes[0])
    break

    
