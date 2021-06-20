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
import decimal
import time

np.random.seed(77)
random.seed(77)
min_obj_size = {
    "length": 1.5, #Meters
    "width": 0.75, #Meters
    "height": 0.5, #Meters
    "points": 10 # Min number of points
}

# Other Necessary Info
config = {
    'splits': {
        'train': 0.7,
        'test': 0.20,
        'val': 0.1
    }
}
raw_data_path = "/dg-hl-fast/codes/OpenPCDet/data/carla/mdls/"
towns = ['town_1', 'town_2']
num_files_in_towns = [None] * len(towns)

def create_folder_structure():
    """
        Creating file structure 
    """
    base_path = raw_data_path + 'as_kitti'
    Path(base_path).mkdir(parents=True, exist_ok=True)
    Path(base_path + "/ImageSets").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/training/calib").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/training/velodyne").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/training/label_2").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/training/image_2").mkdir(parents=True, exist_ok=True)

    Path(base_path + "/testing/calib").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/testing/velodyne").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/testing/label_2").mkdir(parents=True, exist_ok=True)
    Path(base_path + "/testing/image_2").mkdir(parents=True, exist_ok=True)

def create_data_splits():
    """
        Creates ImageSets
    """
    # Collecting files
    num_files = 0
    for i in range(len(towns)):
        town = towns[i]
        path, dirs, files = next(os.walk(raw_data_path + town + '/frame'))
        
        num_files_in_towns[i] = len(files)
        num_files += len(files)

        # num_files_in_towns[i] = 30
        # num_files += 30

    # Splitting the towns data into train, test, val
    files = np.arange(0, num_files)
    np.random.shuffle(files)

    train_split_ind = int(config['splits']['train'] * num_files)
    test_split_ind = train_split_ind + int(config['splits']['test'] * num_files)
    train_splits = files[0: train_split_ind]
    test_splits = files[train_split_ind: test_split_ind]
    val_splits = files[test_split_ind:]

    # train_splits = files[0: 2]
    # test_splits = files[2: 4]
    # val_splits = files[4:6]    

    return train_splits, test_splits, val_splits, num_files

def save_data_splits(splits, split_type):

    image_sets_path = raw_data_path + 'as_kitti/ImageSets'
    
    # Check if already exists -- override ?
    if os.path.isfile(image_sets_path + '/train.txt'):
        print("Dataset splits already exist")

    # Saving the splits
    np.savetxt(image_sets_path + '/' + split_type + '.txt', splits, fmt='%04d')

def convert_global_index_to_local_index(num_files_in_towns, idx):

    aggr_start = 0
    aggr_end = 0
    for i in range(len(num_files_in_towns)):
        aggr_start = aggr_end
        aggr_end = aggr_start + num_files_in_towns[i]        
        if aggr_start <= idx < aggr_end:
            return (i, idx - aggr_start)

def get_dataset_type(idx, splits):

    if np.where(splits[0] == idx)[0].shape[0] > 0:
        return 'train'
    elif np.where(splits[1] == idx)[0].shape[0] > 0:
        return 'test'
    elif np.where(splits[2] == idx)[0].shape[0] > 0:
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

    """
        Parse Calib data
        Fields ( in order ): horiz_offset, vert_offset, rot_correction, vert_correction
    """
    num_rays = lidar_calib_data.shape[0]
    parsed_calib = [None] * num_rays
    for ray in range(num_rays):
        temp = {}
        temp['horiz_offset'] = lidar_calib_data[ray][0]
        temp['vert_offset'] = lidar_calib_data[ray][1]
        temp['rot_correction'] = lidar_calib_data[ray][2]
        temp['vert_correction'] = lidar_calib_data[ray][3]

        parsed_calib[ray] = temp

    return parsed_calib

def is_empty_bbox(x, y, z, l, w, h):

    result = False
    if l == 0 or w == 0 or h == 0:
        result = True

    return result

def has_atleast_minimum_size(l, w, h, num_pts):

    result = True
    if l < min_obj_size["length"]:
        result = False
    elif w < min_obj_size["width"]:
        result = False
    elif h < min_obj_size["height"]:
        result = False
    elif num_pts < min_obj_size["points"]:
        result = False

    return result


def convert_polar_to_cartesian(data, header_info, lidar_calib_data):
    
    """
        Reference: Deserialize state data, lidar data sample code from mdls maual
    """
    # Info of main car
    s_p = data.state.position
    sensor_position = np.array([s_p.x, s_p.y, s_p.z]) # [x, y, z ]
    s_o = data.state.orientation
    sensor_orientation = [s_o.axis.x, s_o.axis.y, s_o.axis.z, s_o.angle] # [o_x, o_y, o_z, angle] ? 

    """
        Converting the point clouds to cartesian
    """
    points_per_channel = header_info.points_count_by_channel
    points = data.points
    transformed_points = np.zeros((len(points), 4)) # No intensity. # Marking it as zero
    point_labels = -1 * np.ones((len(points)))

    for p_idx in range(len(points)):

        curr_point = points[p_idx]
        laser_range = curr_point.range
        rotation = curr_point.rotation * (math.pi / 180)
        
        # Ignore -1 laser returns
        if laser_range > 0:
            """
                The points are organized as 
            """
            laser_id = int(p_idx / points_per_channel)
            curr_laser_data = lidar_calib_data[laser_id]

            cos_vert_angle = math.cos(curr_laser_data['vert_correction'])
            sin_vert_angle = math.sin(curr_laser_data['vert_correction'])

            cos_rot_correction = math.cos(curr_laser_data['rot_correction'])
            sin_rot_correction = math.sin(curr_laser_data['rot_correction'])

            cos_rot_angle = math.cos(rotation) * cos_rot_correction + math.sin(rotation) * sin_rot_correction
            sin_rot_angle = math.sin(rotation) * cos_rot_correction - math.cos(rotation) * sin_rot_correction

            xy_distance = laser_range * cos_vert_angle - curr_laser_data['vert_offset'] * sin_vert_angle

            # Point in meters
            x = xy_distance * cos_rot_angle + curr_laser_data['horiz_offset'] * sin_rot_angle
            y = -xy_distance * sin_rot_angle + curr_laser_data['horiz_offset'] * cos_rot_angle
            z = laser_range * sin_vert_angle + curr_laser_data['vert_offset'] * cos_vert_angle

            transformed_points[p_idx, 0] = x
            transformed_points[p_idx, 1] = y
            transformed_points[p_idx, 2] = z

            point_labels[p_idx] = curr_point.object_id


    """ 
        Raw data has all the objects in carla
        Need to keep only the ones that are in the view
        Ref: https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    """

    # return transformed_points
    # print(header_info.object_ids)
    # object_labels = np.zeros((len(header_info.object_ids), 15))
    object_labels_list = np.zeros((1, 15))
    for obj_id in header_info.object_ids:

        curr_instance_mask = point_labels == obj_id
        curr_instance_points = transformed_points[curr_instance_mask.reshape(-1), :]

        if curr_instance_points.shape[0] == 0:
            continue

        # Getting bounding box coordinates
        x_min = np.min(curr_instance_points[:, 0])
        x_max = np.max(curr_instance_points[:, 0])

        y_min = np.min(curr_instance_points[:, 1])
        y_max = np.max(curr_instance_points[:, 1])

        z_min = np.min(curr_instance_points[:, 2])
        z_max = np.max(curr_instance_points[:, 2])                

        # Getting centers
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        z = (z_min + z_max) / 2

        # Getting extent
        l_t = x_max - x_min
        w_t = y_max - y_min
        h = z_max - z_min

        l = max(l_t, w_t)
        w = min(l_t, w_t)

        # Computing r_y in velodyne coordinates. Will be converted to camera co-ordinates when assigning
        r_y = 0 
        if l == l_t:
            r_y = 0
        else:
            r_y = np.pi / 2

        if not has_atleast_minimum_size(l, w, h, curr_instance_points.shape[0]):
            continue

        # objects = object_labels[obj_id, :]
        object_labels = np.empty((1, 15), dtype='object')
        object_labels[0, 0] = 'Car'
        object_labels[0, 1] = 0 # Truncation. 0 = non-truncated
        object_labels[0, 2] = 0 # Occluded. 0 = fully-visible
        object_labels[0, 3] = 0 # Alpha. observation angle - calculate using centers of this and the observation vehicle
        object_labels[0, 4] = 0 # (4 params) bbox in image. Not needed for our point cloud based method
        object_labels[0, 5] = 0
        object_labels[0, 6] = 0
        object_labels[0, 7] = 0
        object_labels[0, 8] = round(h, 2) # 3d object co-ordinates (h, w, l, x,y,z (3D object location in camera coordinates (in meters)))
        object_labels[0, 9] = round(w, 2)
        object_labels[0, 10] = round(l, 2)
        object_labels[0, 11] = round(x, 2)
        object_labels[0, 12] = round(y, 2)
        object_labels[0, 13] = round(z - round(h, 2)/2, 2) # Kitti's box center is bottom center ( Ref # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/kitti.py)
        object_labels[0, 14] = -(r_y + np.pi / 2) # Converting to camera coordinates from velodyne
        # objects[15] = 0 # Score ?

        object_labels_list = np.vstack([object_labels_list, object_labels])

    return transformed_points, object_labels_list[1:, :], point_labels

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

def generate_caliberation_file():

    """
        Reference: https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
    """
    calib_mat = np.empty((7, 13), dtype="object")
    identity_34 = np.zeros((3, 4))
    eye_33 = np.eye((3))
    identity_34[:3, :3] = eye_33
    identity_34 = identity_34.reshape(-1)

    calib_mat[0, 0] = "P0:"
    calib_mat[0, 1:] = identity_34

    calib_mat[1, 0] = "P1:"
    calib_mat[1, 1:] = identity_34

    calib_mat[2, 0] = "P2:"
    calib_mat[2, 1:] = identity_34

    calib_mat[3, 0] = "P3:"
    calib_mat[3, 1:] = identity_34

    calib_mat[4, 0] = "R0_rect:"
    calib_mat[4, 1:10] = eye_33.reshape(-1)
    calib_mat[4, 10] = ''
    calib_mat[4, 11] = ''
    calib_mat[4, 12] = ''

    calib_mat[5, 0] = "Tr_velo_to_cam:"
    calib_mat[5, 1:] = identity_34 # Camera not involved 

    calib_mat[6, 0] = "Tr_imu_to_velo:"
    calib_mat[6, 1:] = identity_34
    calib_mat[6, 12] = -2

    return calib_mat

def parse_frame_and_save_data(data, global_id, town_idx, idx_in_that_town, header_info, lidar_calib_data, dtype):

    # Convert & save the point cloud
    point_cloud_data, object_labels, point_labels = convert_polar_to_cartesian(data, header_info, lidar_calib_data)

    if object_labels.shape[0] == 0:
        print("No objectes detected", global_id)
        return False

    # Generate the Caliberation file
    calib_data = generate_caliberation_file()

    if dtype == 'test':
        folder = 'testing'
    else:
        folder = 'training'

    global_id = format(global_id, "06d")

    point_cloud_file_path = '%sas_kitti/%s/%s/%s.%s' % (raw_data_path, folder, 'velodyne', global_id, 'npy')
    label_path = '%sas_kitti/%s/%s/%s.%s' % (raw_data_path, folder, 'label_2', global_id, 'txt')
    point_cloud_label_path = '%sas_kitti/%s/%s/%s.%s' % (raw_data_path, folder, 'label_2', global_id + '_label', 'npy')
    calib_data_path = '%sas_kitti/%s/%s/%s.%s' % (raw_data_path, folder, 'calib', global_id, 'txt')

    np.save(point_cloud_file_path, point_cloud_data)
    np.savetxt(label_path, object_labels, fmt='%s')
    # np.save(point_cloud_label_path, point_labels)
    np.savetxt(calib_data_path, calib_data, fmt='%s')

    return True

def convert_dataset_to_kitti_format():

    create_folder_structure()
    header_infos = read_header_infos(towns)
    lidar_calib_data = get_lidar_calib()
    
    splits = create_data_splits()
    num_files = splits[-1]
    splits = splits[:3]

    dtypes = ['train', 'test', 'val']
    # for i in range(num_files):
    # # ind_req = 2059
    # # for i in range(ind_req, ind_req + 1):

    #     # Get meta data of id
    #     dtype = get_dataset_type(i, splits)

    #     # Read that frame
    #     curr_frame_data, town_idx, idx_in_that_town = read_curr_frame_data(i)

    #     # Save the frame in kitti format
    #     parse_and_save_status = parse_frame_and_save_data(curr_frame_data, i, town_idx, idx_in_that_town, header_infos[town_idx], lidar_calib_data, dtype)

    #     print("Processed frame: ", i)
    overall_completed_ctr = 0
    for split_idx in range(3):

        curr_split = splits[split_idx]
        # Get meta data of id
        dtype = dtypes[split_idx]
        num_files_in_split = curr_split.shape[0]
        status = np.empty(num_files_in_split, dtype=bool)
        status[:]=False
        # Read that frame
        for idx in range(num_files_in_split):
            i = curr_split[idx]
            curr_frame_data, town_idx, idx_in_that_town = read_curr_frame_data(i)

            # Save the frame in kitti format
            parse_and_save_status = parse_frame_and_save_data(curr_frame_data, i, town_idx, idx_in_that_town, header_infos[town_idx], lidar_calib_data, dtype)
            
            status[idx] = parse_and_save_status
            overall_completed_ctr += 1
            print("Processed frame: ", overall_completed_ctr, '/', num_files, end='\r')    

        save_data_splits(curr_split[status], dtype)

def find_ranges():

    training_velodyne_path = raw_data_path + 'as_kitti/testing/velodyne/'
    _x = [0, 0]
    _y = [0, 0]
    _z = [0, 0]

    list_of_point_clouds = os.listdir(training_velodyne_path)
    for pc_path in list_of_point_clouds:
        pc = np.load(training_velodyne_path + pc_path)

        possible_min, possible_max = np.min(pc[:, 0]), np.max(pc[:, 0])
        if possible_min < _x[0]:
            _x[0] = possible_min
        if possible_max > _x[1]:
            _x[1] = possible_max

        
        possible_min, possible_max = np.min(pc[:, 1]), np.max(pc[:, 1])
        if possible_min < _y[0]:
            _y[0] = possible_min
        if possible_max > _y[1]:
            _y[1] = possible_max     

        possible_min, possible_max = np.min(pc[:, 2]), np.max(pc[:, 2])
        if possible_min < _z[0]:
            _z[0] = possible_min
        if possible_max > _z[1]:
            _z[1] = possible_max     

    print("The 1st dim range is: ", _x)
    print("The 2nd dim range is: ", _y)
    print("The 3rd dim range is: ", _z)

if __name__ == '__main__':
    
    tic = time.perf_counter()
    convert_dataset_to_kitti_format()
    toc = time.perf_counter()
    print(f"Total Time taken is {toc - tic:0.4f} seconds")

    # Find max ranges
    # find_ranges()