from ultralytics import YOLO
import os
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math   
import time
from sklearn.preprocessing import StandardScaler

robot_height = 0.545  # in meters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detection_for_train(dir, yolo_model):
    ''' Takes the directory of images and returns a dataframe with scaled size of the robot if detected in the image'''
    df = pd.DataFrame(columns=["filename", "width", "height", "size", "parallel_pixel_dist"])
    for image_path in glob.glob(dir):
        #print(image_path)
        result = yolo_model(source=image_path,conf=0.6,save=False)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                xyxy = result.boxes.xyxy.cpu().numpy()
                mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)  
                mid_coord = (result.orig_img.shape[0], result.orig_img.shape[1])   ## scaled size
                pixel_dist = 1000*(mid_robot[0]-mid_coord[0])/result.orig_img.shape[0] 
                #print(xywh)
                w = xywh[0][2]
                h = xywh[0][3]  # 0 added as the first index since only one robot in every frame
                scaled_size = 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1])   ## scaling so that the same model works for images with differenct resolutions   ## multiplied by 10 for scale
                new_row = {"filename": os.path.basename(image_path), "width": w, "height": h, "size": scaled_size, "parallel_pixel_dist": pixel_dist}
                df.loc[len(df)] = new_row
    return df

def detection_for_train_with_uv(dir, yolo_model):
    ''' Takes the directory of images and returns a dataframe with scaled size of the robot if detected in the image'''
    df = pd.DataFrame(columns=["filename", "width", "height", "size", "u", "v","xyxy"])
    T = 0
    for image_path in glob.glob(dir):
        #print(image_path)
        t = time.time()
        result = yolo_model(source=image_path,conf=0.6,save=False)  # predict on an image
        T += time.time() - t
        for result in result:
            if len(result.boxes.xywh)==0:
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                xyxy = result.boxes.xyxy.cpu().numpy()
                u = xywh[0][0]/result.orig_img.shape[1]
                v = (xywh[0][1] + xywh[0][3])/result.orig_img.shape[0]
                #print(xywh)
                w = xywh[0][2]/result.orig_img.shape[1]
                h = xywh[0][3]/result.orig_img.shape[0]  # 0 added as the first index since only one robot in every frame
                scaled_size = w*h  ## scaling so that the same model works for images with differenct resolutions   ## multiplied by 10 for scale
                new_row = {"filename": os.path.basename(image_path), "width": w, "height": h, "size": scaled_size, "u": u, "v": v, "xyxy": xyxy}
                df.loc[len(df)] = new_row
    return df, T


def combine_df_for_train(data_df, gt_df):
    '''Combine the dataframes to add the ground truth perpendicular distances to the data_df'''
    # Loop through rows using iterrows()
    data_df['gt_x'] = None
    data_df['gt_y'] = None
    data_df['gt_heading'] = None
    data_df['abs_dist'] = None
    data_df['angle'] = None
    for size_idx, size_row in data_df.iterrows():
        time_diff = abs(float(gt_df['timestamp'][0]) - float(size_row['filename'].split('_')[0]))
        side = size_row['filename'].split('_')[1].split('.')[0]
        for gt_idx, gt_row in gt_df.iterrows():
            if time_diff >= abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0])):
                time_diff = abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0]))
                idx = gt_idx
            else:
                break
        data_df['gt_x'][size_idx] = gt_df['x'][idx]
        data_df['gt_y'][size_idx] = gt_df['y'][idx]
        data_df['gt_heading'][size_idx] = gt_df['heading'][idx]
        data_df['abs_dist'][size_idx] = (gt_df['x'][idx]**2+gt_df['y'][idx]**2)**0.5
        if side == 'F':
            if float(gt_df['y'][idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(float(gt_df['x'][idx]),-float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now 
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] )
            elif float(gt_df['x'][idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
            # elif float(gt_df['x'][idx])<0:
            #     data_df['angle'][size_idx] = -3.14/2  
        elif side == 'B':
            if float(gt_df['y'][idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(-float(gt_df['x'][idx]),float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] ) 
            elif float(gt_df['x'][idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
            # elif float(gt_df['x'][idx])<0:
            #     data_df['angle'][size_idx] = -3.14/2  
    return data_df


def real_dist_to_robot_from_mid(xyxy, mid_xy, robot_height):
    ''' Function to calculate the perpendicular distance from the middle pixel of the image to the middle pixel of the robot'''
    mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)      ############################################################################################################ Single robot in every frame
    pixel_dist = abs(mid_robot[0]-mid_xy[0]) # pixel_dist = ((mid_robot[0]-mid_xy[0])**2+(mid_robot[1]-mid_xy[1])**2)**0.5   
    h = abs(xyxy[0][1]-xyxy[0][3])
    real_dist = pixel_dist*robot_height/h
    return real_dist


def detection_get_distance(dir, yolo_model, distance_estimation_model):
    ''' detect the robot using the trained neural network and calculates the distance from the camera to the robot '''
    df = pd.DataFrame(columns=["filename","dist_to_the_robot_pred", "detected_or_not","inference_time"])
    for image_path in glob.glob(dir):
        # print(image_path)
        start_time = time.time()
        result = yolo_model(source=image_path,conf=0.6,save=False)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "dist_to_the_robot_pred": None, "detected_or_not": 0, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                # print(xywh)
                w = xywh[0][2]
                h = xywh[0][3]  # 0 added as the first index since only one robot in every frame
                robot_size_scaled = 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1])   ## scaled size
                xyxy = result.boxes.xyxy.cpu().numpy()
                mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)  
                # print(xyxy)
                mid_xy = (result.orig_img.shape[0]/2, result.orig_img.shape[1]/2)   
                real_parallel_dist = real_dist_to_robot_from_mid(xyxy, mid_xy, robot_height)
                # print(real_parallel_dist)
                robot_size_scaled = torch.tensor(robot_size_scaled, dtype=torch.float32).to(device).reshape(-1,1)
                # print(robot_size_scaled)
                with torch.inference_mode():  # inference_mode turns off keeping track of the gradients 
                    perpendicular_dist = distance_estimation_model(robot_size_scaled)  # get the perpendicular distance from the robot size
                perpendicular_dist = perpendicular_dist.cpu().detach().numpy()[0][0]
                # print(perpendicular_dist)
                abs_dist_to_the_robot = (real_parallel_dist**2+perpendicular_dist**2)**0.5
                pixel_dist = 1000*(mid_robot[0]-mid_xy[0])/result.orig_img.shape[0]
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "dist_to_the_robot_pred": abs_dist_to_the_robot, "detected_or_not": 1, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
    return df


def shalihan(dir, yolo_model, distance_estimation_model):
    ''' detect the robot using the trained neural network and calculates the distance from the camera to the robot '''
    df = pd.DataFrame(columns=["filename", "detected_or_not","xyxy"])
    for image_path in glob.glob(dir):
        # print(image_path)
        start_time = time.time()
        result = yolo_model(source=image_path,conf=0.6,save=True)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "detected_or_not": 0, "xyxy": None}
                df.loc[len(df)] = new_row
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                # print(xywh)
                w = xywh[0][2]
                h = xywh[0][3]  # 0 added as the first index since only one robot in every frame
                robot_size = w*h  ## scaled size
                xyxy = result.boxes.xyxy.cpu().numpy()
                mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)  
                # print(xyxy)
                mid_xy = (result.orig_img.shape[0]/2, result.orig_img.shape[1]/2)   
                real_parallel_dist = real_dist_to_robot_from_mid(xyxy, mid_xy, robot_height)
                # print(real_parallel_dist)
                #robot_size_scaled = torch.tensor(robot_size_scaled, dtype=torch.float32).to(device).reshape(-1,1)
                # print(robot_size_scaled)
                # with torch.inference_mode():  # inference_mode turns off keeping track of the gradients 
                #     perpendicular_dist = distance_estimation_model(robot_size_scaled)  # get the perpendicular distance from the robot size
                # perpendicular_dist = perpendicular_dist.cpu().detach().numpy()[0][0]
                # print(perpendicular_dist)
                # abs_dist_to_the_robot = (real_parallel_dist**2+perpendicular_dist**2)**0.5
                # pixel_dist = 1000*(mid_robot[0]-mid_xy[0])/result.orig_img.shape[0]
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "detected_or_not": 1, "xyxy": xyxy}
                df.loc[len(df)] = new_row
    return df



def detection_get_distance_checkkkkkkkkkk(dir, yolo_model, distance_estimation_model):
    ''' detect the robot using the trained neural network and calculates the distance from the camera to the robot '''
    df = pd.DataFrame(columns=["filename", "size" ,"parallel_pixel_dist","perp_dist_to_the_robot_pred", "detected_or_not","inference_time"])
    for image_path in glob.glob(dir):
        # print(image_path)
        start_time = time.time()
        result = yolo_model(source=image_path,conf=0.6,save=False)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "size": None ,"parallel_pixel_dist": None, "perp_dist_to_the_robot_pred": None, "detected_or_not": 0, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                # print(xywh)
                w = xywh[0][2]
                h = xywh[0][3]  # 0 added as the first index since only one robot in every frame
                robot_size_scaled = 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1])   ## scaled size
                xyxy = result.boxes.xyxy.cpu().numpy()
                mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)  
                # print(xyxy)
                mid_xy = (result.orig_img.shape[0]/2, result.orig_img.shape[1]/2)   
                real_parallel_dist = 0                                                  #real_dist_to_robot_from_mid(xyxy, mid_xy, robot_height)
                # print(real_parallel_dist)
                robot_size_scaled = torch.tensor(robot_size_scaled, dtype=torch.float32).to(device).reshape(-1,1)
                # print(robot_size_scaled)
                with torch.inference_mode():  # inference_mode turns off keeping track of the gradients 
                    perpendicular_dist = distance_estimation_model(robot_size_scaled)  # get the perpendicular distance from the robot size
                perpendicular_dist = perpendicular_dist.cpu().detach().numpy()[0][0]
                # print(perpendicular_dist)
                abs_dist_to_the_robot = (real_parallel_dist**2+perpendicular_dist**2)**0.5
                pixel_dist = 1000*(mid_robot[0]-mid_xy[0])/result.orig_img.shape[0]
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "size": 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1]), "parallel_pixel_dist":pixel_dist , "perp_dist_to_the_robot_pred": abs_dist_to_the_robot, "detected_or_not": 1, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
    return df


def detection_get_distance_and_angle(dir, yolo_model, distance_estimation_model):
    ''' detect the robot using the trained neural network and calculates the distance from the camera to the robot '''
    df = pd.DataFrame(columns=["filename", "size" ,"parallel_pixel_dist","perp_dist_to_the_robot_pred","angle_pred", "detected_or_not","inference_time"])
    for image_path in glob.glob(dir):
        # print(image_path)
        start_time = time.time()
        result = yolo_model(source=image_path,conf=0.6,save=True)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "size": None ,"parallel_pixel_dist": None, "perp_dist_to_the_robot_pred": None,"angle_pred": None, "detected_or_not": 0, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                # print(xywh)
                w = xywh[0][2]
                h = xywh[0][3]  # 0 added as the first index since only one robot in every frame
                robot_size_scaled = 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1])   ## scaled size
                xyxy = result.boxes.xyxy.cpu().numpy()
                mid_robot = ((xyxy[0][0]+xyxy[0][2])/2, (xyxy[0][1]+xyxy[0][3])/2)  
                # print(xyxy)
                mid_xy = (result.orig_img.shape[0]/2, result.orig_img.shape[1]/2)   
                real_parallel_dist = 0                                                  #real_dist_to_robot_from_mid(xyxy, mid_xy, robot_height)
                # print(real_parallel_dist)
                robot_size_scaled = torch.tensor(robot_size_scaled, dtype=torch.float32).to(device).reshape(-1,1)
                parallel_pixel_dist = 1000*(mid_robot[0]-mid_xy[0])/result.orig_img.shape[0]
                input = torch.tensor([robot_size_scaled, parallel_pixel_dist], dtype=torch.float32).to(device).reshape(-1, 2)
                # print(robot_size_scaled)
                with torch.inference_mode():  # inference_mode turns off keeping track of the gradients 
                    output = distance_estimation_model(input)  # get the perpendicular distance from the robot size
                perpendicular_dist = output.cpu().detach().numpy()[0][0]
                angle_pred = output.cpu().detach().numpy()[0][1]
                # print(perpendicular_dist)
                abs_dist_to_the_robot = (real_parallel_dist**2+perpendicular_dist**2)**0.5
                end_time = time.time()
                elapsed_time = end_time - start_time
                new_row = {"filename": os.path.basename(image_path), "size": 10000*w*h/(result.orig_img.shape[0]*result.orig_img.shape[1]), "parallel_pixel_dist":parallel_pixel_dist, "perp_dist_to_the_robot_pred": abs_dist_to_the_robot, "angle_pred":angle_pred, "detected_or_not": 1, "inference_time": elapsed_time*1000}
                df.loc[len(df)] = new_row
    return df

def detection_get_distance_and_angle_with_uv(dir, yolo_model, distance_estimation_model, angle_estimation_model):
    ''' detect the robot using the trained neural network and calculates the distance from the camera to the robot '''
    df = pd.DataFrame(columns=["filename","abs_dist_pred","angle_pred", "detected_or_not"])
    for image_path in glob.glob(dir):
        # print(image_path)
        result = yolo_model(source=image_path,conf=0.6,save=False)  # predict on an image
        for result in result:
            if len(result.boxes.xywh)==0:
                new_row = {"filename": os.path.basename(image_path), "abs_dist_pred": None,"angle_pred": None, "detected_or_not": 0}
                df.loc[len(df)] = new_row
                break  
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                u = xywh[0][0]/result.orig_img.shape[1]
                v = (xywh[0][1] + xywh[0][3])/result.orig_img.shape[0]
                #print(xywh)
                w = xywh[0][2]/result.orig_img.shape[1]
                h = xywh[0][3]/result.orig_img.shape[0]  # 0 added as the first index since only one robot in every frame
                input_dist = torch.tensor([u, v, w, h], dtype=torch.float32).to(device).reshape(-1, 4)
                input_angle = torch.tensor([u, v, w, h], dtype=torch.float32).to(device).reshape(-1, 4)
                # print(robot_size_scaled)
                with torch.inference_mode():  # inference_mode turns off keeping track of the gradients 
                    abs_dist_pred = distance_estimation_model(input_dist)  # get the perpendicular distance from the robot size
                    angle_pred = angle_estimation_model(input_angle) 
                abs_dist_pred  = abs_dist_pred.cpu().detach().numpy()[0][0]
                angle_pred = angle_pred.cpu().detach().numpy()[0][0]
                new_row = {"filename": os.path.basename(image_path), "abs_dist_pred": abs_dist_pred, "angle_pred":angle_pred, "detected_or_not": 1}
                df.loc[len(df)] = new_row
    return df


def process_df(data_df, data_idx, data_row, time_split_1, time_split_2):
    ''' adds the robot presence to the dataframe '''
    side = data_row['filename'].split('_')[1].split('.')[0]
    time = float(data_row['filename'].split('_')[0])
    if side == 'F' and time < time_split_1:
        data_df['robot_in_the_frame_or_not'][data_idx] = 1
    elif side == 'B' and time_split_2 > time > time_split_1:
        data_df['robot_in_the_frame_or_not'][data_idx] = 1
    elif side == 'F' and time > time_split_2:
        data_df['robot_in_the_frame_or_not'][data_idx] = 1 
    else: 
        data_df['robot_in_the_frame_or_not'][data_idx] = 0


def combine_df(data_df, gt_df, time_split_1, time_split_2):
    '''Combine the dataframes to add the abs distances to the data_df'''
    # Loop through rows using iterrows()
    data_df['robot_in_the_frame_or_not'] = None
    data_df['abs_dist_gt'] = None
    data_df['angle'] = None
    for size_idx, size_row in data_df.iterrows():
        print(size_idx)
        time_diff = abs(float(gt_df['timestamp'][0]) - float(size_row['filename'].split('_')[0]))
        side = size_row['filename'].split('_')[1].split('.')[0]
        process_df(data_df ,size_idx, size_row, time_split_1, time_split_2)
        for gt_idx, gt_row in gt_df.iterrows():
            if time_diff >= abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0])):
                idx = gt_idx
                time_diff = abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0]))
                
            else:
                break
        data_df['abs_dist_gt'][size_idx] = (gt_df['x'][idx]**2+gt_df['y'][idx]**2)**0.5
        if side == 'F':
            if float(gt_df['y'][idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(float(gt_df['x'][idx]),-float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now 
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] ) 
            elif float(gt_df['x'][idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
        elif side == 'B':
            if float(gt_df['y'][idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(-float(gt_df['x'][idx]),float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now 
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] ) 
            elif float(gt_df['x'][idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
    return data_df



def ran(data_df, gt_df, time_split_1, time_split_2):
    '''Combine the dataframes to add the abs distances to the data_df'''
    # Loop through rows using iterrows()
    data_df['robot_in_the_frame_or_not'] = None
    data_df['gt_x'] = None
    data_df['gt_y'] = None
    data_df['gt_heading'] = None
    for size_idx, size_row in data_df.iterrows():
        print(size_idx)
        time_diff = abs(float(gt_df['timestamp'][0]) - float(size_row['filename'].split('_')[0]))
        side = size_row['filename'].split('_')[1].split('.')[0]
        process_df(data_df ,size_idx, size_row, time_split_1, time_split_2)
        for gt_idx, gt_row in gt_df.iterrows():
            if time_diff >= abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0])):
                idx = gt_idx
                time_diff = abs(float(gt_row['timestamp']) - float(size_row['filename'].split('_')[0]))
                
            else:
                break
        data_df['gt_x'][size_idx] = gt_df['x'][idx]
        data_df['gt_y'][size_idx] = gt_df['y'][idx]
        data_df['gt_heading'][size_idx] = gt_df['heading'][idx]
        # if side == 'F':
        #     if float(gt_df['y'][idx])!=float(0):
        #         data_df['angle'][size_idx] = math.atan2(float(gt_df['x'][idx]),-float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now 
        #     elif float(gt_df['x'][idx])>0:
        #         data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
        #     elif float(gt_df['x'][idx])<0:
        #         data_df['angle'][size_idx] = -3.14/2  
        # elif side == 'B':
        #     if float(gt_df['y'][idx])!=float(0):
        #         data_df['angle'][size_idx] = math.atan2(-float(gt_df['x'][idx]),float(gt_df['y'][idx]))   # taninverse(x/-y)   normal axis system now 
        #     elif float(gt_df['x'][idx])>0:
        #         data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
        #     elif float(gt_df['x'][idx])<0:
        #         data_df['angle'][size_idx] = -3.14/2     
    return data_df

def avg_error_calc(data_df_with_gt):
    ''' calculate the mean absolute error if a robot is detected in the image'''
    abs_errors = []
    count = 0
    for idx, row in data_df_with_gt.iterrows():
        if row['robot_in_the_frame_or_not'] == 1 and row['detected_or_not'] == 1:
            count += 1
            abs_errors.append(abs(row['dist_to_the_robot_gt']-row['dist_to_the_robot_pred']))
    return sum(abs_errors)/count



def calc_abs_dist_and_angle(data_df):
    data_df['abs_dist'] = None
    data_df['angle'] = None
    for size_idx, size_row in data_df.iterrows():
        side = size_row['filename'].split('_')[1].split('.')[0]
        data_df['abs_dist'][size_idx] = (data_df['gt_x'][size_idx]**2+data_df['gt_y'][size_idx]**2)**0.5
        if side == 'F':
            if float(data_df['gt_y'][size_idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(float(data_df['gt_x'][size_idx]),-float(data_df['gt_y'][size_idx]))   # taninverse(x/-y)   normal axis system now 
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] )
            elif float(data_df['gt_x'][size_idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
            # elif float(gt_df['x'][idx])<0:
            #     data_df['angle'][size_idx] = -3.14/2  
        elif side == 'B':
            if float(data_df['gt_y'][size_idx])!=float(0):
                data_df['angle'][size_idx] = math.atan2(-float(data_df['gt_x'][size_idx]),float(data_df['gt_y'][size_idx]))   # taninverse(x/-y)   normal axis system now
                if data_df['angle'][size_idx]<0:
                    data_df['angle'][size_idx] = math.pi + ( math.pi + data_df['angle'][size_idx] ) 
            elif float(data_df['gt_x'][size_idx])>0:
                data_df['angle'][size_idx] = 3.14/2    #####   check this ####################################################################
            # elif float(gt_df['x'][idx])<0:
            #     data_df['angle'][size_idx] = -3.14/2  
    return data_df


def calc_uvwh(data_df):
    data_df['u'] = None
    data_df['v'] = None
    data_df['w'] = None
    data_df['h'] = None
    for idx, row in data_df.iterrows():
        u = (row['x1']+row['x2'])/2
        v = (row['y1']+row['y2'])/2
        w = row['x2']-row['x1']
        h = row['y2']-row['y1']
        data_df['u'][idx] = u
        data_df['v'][idx] = v
        data_df['w'][idx] = w
        data_df['h'][idx] = h
    
    return data_df

def standard_scale(data_df):
    cols_to_scale = ['u', 'v', 'w', 'h', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X3', 'Y3', 'Z3', 'X4', 'Y4', 'Z4']
    
    data_scaled = data_df[cols_to_scale]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    data_df[cols_to_scale] = data_scaled

    return data_df
