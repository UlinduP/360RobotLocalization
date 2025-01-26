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
from time import time

device = torch.device("cpu")
robot_height = 0.545 # m
torch.manual_seed(0)

time_split_1 = 1809285220.843473  # 2024-02-06-21-40-49 time_split_1 = 1707227007.194655 , time_split_2 = 1707227126.116658  ,,    2024-02-06-21-34-18    time_split_1 = 1707226624.239907, time_split_2 = 1707226750.236465 ,, 
time_split_2 = 1819285220.290048  # 2024-02-20-16-34-57 time_split_1 = 1708418363.176522 , time_split_2 = 1708418501.309361 ,  2024-02-20-16-44-28 time_split_1 = 1708418913.169285, time_split_2 = 1708419046.290048

#Training_Results/insta360_train1/2048X2048imgs_night_epochs=300
model = YOLO("Training_Results/insta360_train1/combined_dataset/weights/best.pt")  # load a pretrained model (recommended for training)  D:\Ulindu\turtlebot_detection\Training_Results\insta360_train1\HR_First_Floor_2048X2048\weights\best.pt
# Move the model to the GPU if available      
model.to(device)
print(model.device)

# train_dir = f'insta360/2024-02-20-16-34-57_images/*.jpg'       
# test_dir = f'insta360/2024-02-20-16-44-28_images/*.jpg'             

# train_df = utils.detection_for_train(train_dir, yolo_model=model)
# test_df = utils.detection_for_train(test_dir, yolo_model=model)

# column_names = ['timestamp', 'x', 'y','heading']
# train_gt = pd.read_csv('./insta360/2024-02-20-16-34-57_gt.txt',sep=',',names=column_names)
# test_gt = pd.read_csv('./insta360/2024-02-20-16-44-28_gt.txt',sep=',',names=column_names)

# train_df_with_gt = utils.combine_df_for_train(train_df, train_gt)
# test_df_with_gt = utils.combine_df_for_train(test_df, test_gt)

# train_df_with_gt.to_csv("./csv_data/2024-02-20-16-34-57_first_floor_training_dataframe.csv", index=False)
# test_df_with_gt.to_csv("./csv_data/2024-02-20-16-44-28_first_floor_training_dataframe.csv", index=False)

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


## These are dataframes without the perpendicular distances
#data_dir = 'insta360_renamed_2024_01_31/training_rename_dataset_2024-01-31-17-27-34/*.jpg'     insta360/training_2024-02-06-21-34-18_images
data_dir = f'/media/hariharan/ULINDU/Ulindu/turtlebot_detection/insta360/2024-03-12-14-50-28_images/*.jpg'        #insta360\validation_2024-02-06-21-40-49_images       insta360/2024-02-20-16-34-57_images

column_names = ['timestamp', 'x', 'y','heading']
gt_df = pd.read_csv('/media/hariharan/ULINDU/Ulindu/turtlebot_detection/insta360/2024-03-12-14-50-28_gt.txt',sep=' ',names=column_names)

distance_estimation_model = distance_estimation.Distance3()
distance_estimation_model.load_state_dict(torch.load("/media/hariharan/ULINDU/Ulindu/turtlebot_detection/models/Distance3.pth", map_location=torch.device('cpu')))  # models/model_1-8-16-32-64-32-16-8-1_Night_2048X2048imgs.pth      #\model_1-8-16-32-64-32-16-8-1_2048X2048imgs_first_floor.pth
distance_estimation_model.to(device)  

angle_estimation_model = distance_estimation.Angle1()
angle_estimation_model.load_state_dict(torch.load("/media/hariharan/ULINDU/Ulindu/turtlebot_detection/models/Angle1_THE_BEST.pth", map_location=torch.device('cpu')))
angle_estimation_model.to(device)

# data_df , T= utils.detection_for_train_with_uv(data_dir, yolo_model=model)  
# data_df_with_gt = utils.ran(data_df, gt_df, time_split_1 = time_split_1, time_split_2 = time_split_2)        ##### Night HR Testing datset ( 1707227007.194655  1707227126.116658  )      
#### first floor ( time_split_1 = 1708418913.169285, time_split_2 = 1708419046.290048 )

# change the order of columns
# data_df_with_gt = data_df_with_gt[['filename', 'robot_in_the_frame_or_not','detected_or_not','gt_x','gt_y','gt_heading','xyxy']]

# plot.plot_image_number_vs_dist(data_df_with_gt)

# data_df_with_gt.to_csv("./csv_data/ran_2024-03-01-17-27-00.csv", index=False)

t = time()
result = model(source='/media/hariharan/ULINDU/Ulindu/turtlebot_detection/insta360/2024-03-12-14-50-28_images/1710226228.991394_F.jpg',conf=0.6,save=True)
T = time()-t

print(T)



