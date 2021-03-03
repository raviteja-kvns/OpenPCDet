import mdlidar_pb2

# Reading the files
curr_frame = mdlidar_pb2.Frame()
print("Successfully read frame object", curr_frame)
print(dir(curr_frame))

with open("/dg-hl-fast/codes/OpenPCDet/data/carla/mdls/town_1/frame/data_1", 'rb') as f:
    curr_frame.ParseFromString(f.read())
    print("The start time is : ", curr_frame.start_time)