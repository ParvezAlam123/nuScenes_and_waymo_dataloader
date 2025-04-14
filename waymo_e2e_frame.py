import tensorflow as tf
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import numpy as np
import cv2 

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2 

filename = '/home/parvez/Downloads/training_202503292338.tfrecord-00000-of-00315' 

dataset = tf.data.TFRecordDataset(filename, compression_type='')  # No compression 

bytes_example = next(dataset_iter)  # Get the next frame (e.g., timestamp)
data = wod_e2ed_pb2.E2EDFrame()
data.ParseFromString(bytes_example)  

# Set the number of frames you want to process
num_frames_to_process = 50

# Loop through multiple frames and process them
for i in range(num_frames_to_process):
    bytes_example = next(dataset_iter)  # Get the next frame (e.g., timestamp)
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(bytes_example) 
    print("time_stamp=", data.frame.timestamp_micros)

    # Extract the desired data for each frame (velocity, acceleration, intent, cameras, etc.)
    if data.HasField('future_states'):
        future_vel_x = data.future_states.vel_x
        future_vel_y = data.future_states.vel_y
        future_accel_x = data.future_states.accel_x
        future_accel_y = data.future_states.accel_y

        print(f"Frame {i+1} - Future Velocity (m/s): X:", future_vel_x, "Y:", future_vel_y)
        print(f"Frame {i+1} - Future Acceleration (m/s^2): X:", future_accel_x, "Y:", future_accel_y)

    if data.HasField('past_states'):
        past_vel_x = data.past_states.vel_x
        past_vel_y = data.past_states.vel_y
        past_accel_x = data.past_states.accel_x
        past_accel_y = data.past_states.accel_y

        print(f"\nFrame {i+1} - Past Velocity (m/s): X:", past_vel_x, "Y:", past_vel_y)
        print(f"Frame {i+1} - Past Acceleration (m/s^2): X:", past_accel_x, "Y:", past_accel_y)

    # Extract and print the driving intent (e.g., go straight, left, right)
    if data.HasField('intent'):
        intent = data.intent
        if intent == wod_e2ed_pb2.EgoIntent.Intent.GO_STRAIGHT:
            print(f"Frame {i+1} - Driving intent: GO STRAIGHT")
        elif intent == wod_e2ed_pb2.EgoIntent.Intent.GO_LEFT:
            print(f"Frame {i+1} - Driving intent: GO LEFT")
        elif intent == wod_e2ed_pb2.EgoIntent.Intent.GO_RIGHT:
            print(f"Frame {i+1} - Driving intent: GO RIGHT")
        else:
            print(f"Frame {i+1} - Driving intent: UNKNOWN")
    
    # Extract images and calibrations from the front cameras
    front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)

    # Concatenate images to show all three front cameras' images side-by-side
    concatenated_image = np.concatenate(front3_camera_image_list, axis=1)

    # Visualize the images
    plt.figure(figsize=(20, 20))
    plt.imshow(concatenated_image)
    plt.title(f"Frame {i+1} - Front Cameras")
    plt.show() 
    
    
    
    
