"""
    This file aims to convert the Carla's MDLS dataset 
    into 
    kitti format
    References: 
    https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
"""
import src_py.mdlidar_pb2 as mdls_reader
from pathlib import Path
import random
import numpy as np
import os
import math

np.random.seed(77)
random.seed(77)

# Other Necessary Info
config = {
    'splits': {
        'train': 0.7,
        'test': 0.15,
        'val': 0.15
    }
}
raw_data_path = "../"

# Creating file structure
base_path = "../as_kitti"
Path(base_path).mkdir(parents=True, exist_ok=True)
Path(base_path + "/ImageSets").mkdir(parents=True, exist_ok=True)
Path(base_path + "/training/calib").mkdir(parents=True, exist_ok=True)
Path(base_path + "/training/velodyne").mkdir(parents=True, exist_ok=True)
Path(base_path + "/training/label_2").mkdir(parents=True, exist_ok=True)
Path(base_path + "/training/image_2").mkdir(parents=True, exist_ok=True)

Path(base_path + "/testing/calib").mkdir(parents=True, exist_ok=True)
Path(base_path + "/testing/velodyne").mkdir(parents=True, exist_ok=True)
Path(base_path + "/testing/image_2").mkdir(parents=True, exist_ok=True)

towns = ['town_1', 'town_2']
num_files_in_towns = [None] * len(towns)
# Create ImageSets
num_files = 0
for i in range(len(towns)):
    town = towns[i]
    path, dirs, files = next(os.walk("../" + town + '/frame'))
    num_files_in_towns[i] = len(files)
    num_files += len(files)

# Splitting the towns data into train, test, val
files = np.arange(0, num_files)
np.random.shuffle(files)

train_split_ind = int(config['splits']['train'] * num_files)
test_split_ind = train_split_ind + int(config['splits']['test'] * num_files)
train_splits = files[0: train_split_ind]
test_splits = files[train_split_ind: test_split_ind]
val_splits = files[test_split_ind:]

np.savetxt('../as_kitti/ImageSets/train.txt', train_splits, fmt='%04d')
np.savetxt('../as_kitti/ImageSets/test.txt', test_splits, fmt='%04d')
np.savetxt('../as_kitti/ImageSets/val.txt', val_splits, fmt='%04d')

def convert_global_index_to_local_index(num_files_in_towns, idx):

    aggr_start = 0
    aggr_end = 0
    for i in range(len(num_files_in_towns)):
        aggr_start = aggr_end
        aggr_end = aggr_start + num_files_in_towns[i]        
        if aggr_start <= idx < aggr_end:
            return (i, idx - aggr_start)

def get_dataset_type(idx):

    if np.where(train_splits == idx)[0].shape[0] > 0:
        return 'train'
    elif np.where(test_splits == idx)[0].shape[0] > 0:
        return 'test'
    elif np.where(val_splits == idx)[0].shape[0] > 0:
        return 'val'
    else:
        return None

def read_curr_frame_data(idx):

    (town_idx, idx_in_that_town) = convert_global_index_to_local_index(num_files_in_towns, idx)
    town_path = raw_data_path + 'town_' + str(town_idx + 1)
    file_path = town_path + '/frame/data_' + str(idx_in_that_town)

    curr_frame = None
    with open(file_path, 'rb') as f:
        curr_frame = mdls_reader.Frame()
        curr_frame.ParseFromString(f.read())

    return curr_frame, town_idx, idx_in_that_town

def get_lidar_calib():
    
    lidar_calib_path = raw_data_path + "raws/vel_64_calib.csv"
    lidar_calib_data = np.genfromtxt(lidar_calib_path, delimiter=',')    

    return lidar_calib_data

def convert_polar_to_cartesian(data, header_info):
    
    """
        Reference: Deserialize state data, lidar data sample code from mdls maual
    """
    # Info of main car
    s_p = data.state.position
    sensor_position = np.array([s_p.x, s_p.y, s_p.z]) # [x, y, z ]
    s_o = data.state.orientation
    sensor_orientation = [s_o.axis.x, s_o.axis.y, s_o.axis.z, s_o.angle] # [x, y, z, angle]

    """
        Converting the point clouds to carteesian
    """
    points_per_channel = header_info.points_count_by_channel
    points = data.points
    transformed_points = np.array((3, len(points)))
    for p_idx in range(len(points)):

        curr_point = points[p_idx]
        range = curr_point.range
        rotation = curr_point.rotation * (math.pi / 180)
        
        if range > 0:
            laser_id = i / points_per_channel

            cos_vert_angle = math.cos(lidar_calib_data[laser_id].vert_correction)
            sin_vert_angle = math.sin(lidar_calib_data[laser_id].vert_correction)

            cos_rot_correction = math.cos(lidar_calib_data[laser_id].rot_correction)
            sin_rot_correction = math.sin(lidar_calib_data[laser_id].rot_correction)

            cos_rot_angle = cos(rotation) * cos_rot_correction + sin(rotation) * sin_rot_correction
            sin_rot_angle = sin(rotation) * cos_rot_correction - cos(rotation) * sin_rot_correction

            xy_distance = range * cos_vert_angle - lidar_calib_data[laser_id].vert_offset * sin_vert_angle

            # Point in meters
            x = xy_distance * cos_rot_angle + lidar_calib_data[laser_id].horiz_offset * sin_rot_angle
            y = -xy_distance * sin_rot_angle + lidar_calib_data[laser_id].horiz_offset * cos_rot_angle
            z = rangee * sin_vert_angle + lidar_calib_data[laser_id].vert_offset * cos_vert_angle

            transformed_points[p_idx, 0] = x
            transformed_points[p_idx, 1] = y
            transformed_points[p_idx, 2] = z

    """ 
        Raw data has all the objects in carla
        Need to keep only the ones that are in the view
        Ref: https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    """
    num_objects = len(data.object_state)
    objects = np.array((num_objects, 15), dtype='o')
    for i in range(num_objects):

        curr_object = data.object_state[i]
        objects[0] = 'Car'
        objects[1] = 0 # Truncation. 0 = non-truncated
        objects[2] = 0 # Occluded. 0 = fully-visible
        objects[3] = 0 # Alpha. observation angle - calculate using centers of this and the observation vehicle
        objects[4] = 0 # (4 params) bbox in image. Not needed for our point cloud based method
        objects[5] = 0
        objects[6] = 0
        objects[7] = 0
        objects[8] = 0 # 3d object co-ordinates (h, w, l, x,y,z (3D object location in camera coordinates (in meters)))
        objects[9] = 0
        objects[10] = 0
        objects[11] = 0
        objects[12] = 0
        objects[13] = 0
        objects[14] = 0 # rotation_y. Get if possible; o/w set 0 as we target axis aligned bbox
        # objects[15] = 0 # Score ?

def read_header_infos(towns):

    header_infos = []
    for town in towns:
        town_path = raw_data_path + town
        file_path = town_path + '/header'

        curr_header = None
        with open(file_path, 'rb') as f:
            curr_header = mdls_reader.Header()
            curr_header.ParseFromString(f.read())
            header_infos.append(curr_header)

    return header_infos


def parse_frame_and_save_data(data, town_idx, idx_in_that_town, header_info, lidar_calib_data, dtype):

    point_cloud = convert_polar_to_cartesian(data, header_info, lidar_calib_data)
    pass

def convert_dataset_to_kitti_format():

    header_infos = read_header_infos(towns)
    lidar_calib_data = get_lidar_calib()

    # for i in range(num_files):
    for i in range(1):

        # Find which dataset it belongs to
        dtype = get_dataset_type(i)
        curr_frame_data, town_idx, idx_in_that_town = read_curr_frame_data(i)
        parse_frame_and_save_data(curr_frame_data, town_idx, idx_in_that_town, header_infos[town_idx], lidar_calib_data, dtype)

if __name__ == '__main__':
    convert_dataset_to_kitti_format()