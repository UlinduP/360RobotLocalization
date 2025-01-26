from ultralytics import YOLO
import os
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.optim as optim
import plot
import distance_estimation
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
robot_height = 0.545 # m
torch.manual_seed(0)

time_split_1 = 1708418913.169285  # 2024-02-06-21-40-49 time_split_1 = 1707227007.194655 , time_split_2 = 1707227126.116658  ,,    2024-02-06-21-34-18    time_split_1 = 1707226624.239907, time_split_2 = 1707226750.236465 ,, 
time_split_2 = 1708419046.290048  # 2024-02-20-16-34-57 time_split_1 = 1708418363.176522 , time_split_2 = 1708418501.309361 ,  2024-02-20-16-44-28 time_split_1 = 1708418913.169285, time_split_2 = 1708419046.290048

#Training_Results/insta360_train1/2048X2048imgs_night_epochs=300
model = YOLO("Training_Results/insta360_train1/combined_dataset/weights/best.pt")  # load a pretrained model (recommended for training)  D:\Ulindu\turtlebot_detection\Training_Results\insta360_train1\HR_First_Floor_2048X2048\weights\best.pt
# Move the model to the GPU if available      
model.to(device)
print(model.device)

train_dir = f'insta360/validation_2024-02-06-21-40-49_images/*.jpg'       
# test_dir = f'insta360/2024-02-20-16-44-28_images/*.jpg'             

train_df = utils.detection_for_train_with_uv(train_dir, yolo_model=model)
# test_df = utils.detection_for_train_with_uv(test_dir, yolo_model=model)

column_names = ['timestamp', 'x', 'y','heading']
train_gt = pd.read_csv('./insta360/2024-02-06-21-40-49_gt.txt',sep=' ',names=column_names)
# test_gt = pd.read_csv('./insta360/2024-02-20-16-44-28_gt.txt',sep=',',names=column_names)

train_df_with_gt = utils.combine_df_for_train(train_df, train_gt)
# test_df_with_gt = utils.combine_df_for_train(test_df, test_gt)
train_df_with_gt = train_df_with_gt[['filename','gt_x','gt_y','gt_heading','xyxy','width','height','size','u','v','abs_dist','angle']]

train_df_with_gt.to_csv("./csv_data/2024-02-06-21-40-49vvvv.csv", index=False)
# test_df_with_gt.to_csv("./csv_data/2024-02-20-16-44-28_first_floor_testing_dataframe.csv", index=False)

# model_0 = distance_estimation.Distance()
# model_0.to(device)
# print(model_0.state_dict())
# loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model_0.parameters(), lr=0.01)

# x_train = train_df_with_gt['size'][:-1]
# y_train = abs(train_df_with_gt['perpendicular_dist'][:-1])

# x_test = test_df_with_gt['size'][:-1]
# y_test = abs(test_df_with_gt['perpendicular_dist'][:-1])

# x_train = torch.tensor(x_train, dtype=torch.float32).to(device).reshape(-1,1)
# y_train = torch.tensor(y_train, dtype=torch.float32).to(device).reshape(-1,1)

# x_test = torch.tensor(x_test, dtype=torch.float32).to(device).reshape(-1,1)   
# y_test = torch.tensor(y_test, dtype=torch.float32).to(device).reshape(-1,1)

# distance_estimation.train(model_0, x_train, y_train, x_test, y_test, loss_fn, optimizer, epochs=1000)

#######################################################################################

# torch.save(model_0.state_dict(), "./models/model_1-8-16-32-64-32-16-8-1_2048X2048imgs_first_floor.pth")