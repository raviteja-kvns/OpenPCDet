#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include "mdlidar.pb.h"

struct LaserCalib {
  float horiz_offset;
  float vert_offset;
  float rot_correction;
  float vert_correction;
};

int main(int argc, char *argv[]) {

  std::string base_file_path = "/path/to/data/";
  std::string calib_file_path = "/path/to/calib-file/";
  
  // Parse calibration file
  std::vector<LaserCalib> calib;
  std::ifstream in(calib_file_path + "vel_64_calib.csv");
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream sep(line);
    std::string field;
    LaserCalib temp;

    std::getline(sep, field, ',');
    temp.horiz_offset = std::stof(field);
    std::getline(sep, field, ',');
    temp.vert_offset = std::stof(field);
    std::getline(sep, field, ',');
    temp.rot_correction = std::stof(field);
    std::getline(sep, field, ',');
    temp.vert_correction = std::stof(field);

    calib.push_back(temp);
  }
  
  // Deserialize single data file
  int data_id = 7;  // file to deserialize is data_7 (arbitrary example)
  std::fstream input(base_file_path + "data_" 
      + std::to_string(data_id), std::ios::in | std::ios::binary);
  mdlidar::Frame data;
  data.ParseFromIstream(&input);

  // Deserialize header data file
  std::fstream input2(base_file_path + "../header",
      std::ios::in | std::ios::binary);
  mdlidar::Header header;
  header.ParseFromIstream(&input2);

  // Process raw lidar data to a local pointcloud
  uint points_per_channel = header.points_count_by_channel();
  int64_t start_time = data.start_time();
  int64_t end_time = data.end_time();
  
  for (int i = 0; i < data.points_size(); ++i) {
    float range = data.points(i).range();
    float rotation = data.points(i).rotation()*M_PI/180.0;  // convert to rad
    
    // Ignore -1 laser returns
    if (range > 0) {
      int laser_id = i/points_per_channel;
      float cos_vert_angle = cos(calib[laser_id].vert_correction);
      float sin_vert_angle = sin(calib[laser_id].vert_correction);
      float cos_rot_correction = cos(calib[laser_id].rot_correction);
      float sin_rot_correction = sin(calib[laser_id].rot_correction);
      
      float cos_rot_angle = 
          cos(rotation)*cos_rot_correction + sin(rotation)*sin_rot_correction;
      float sin_rot_angle = 
          sin(rotation)*cos_rot_correction - cos(rotation)*sin_rot_correction;
      
      float xy_distance = 
          range*cos_vert_angle - calib[laser_id].vert_offset*sin_vert_angle;
          
      // Point coordinates in meters
      float x = xy_distance*cos_rot_angle + calib[laser_id].horiz_offset*sin_rot_angle;
      float y = -xy_distance*sin_rot_angle + calib[laser_id].horiz_offset*cos_rot_angle;
      float z = range*sin_vert_angle + calib[laser_id].vert_offset*cos_vert_angle;
    
      // Time in nanoseconds
      int64_t time = 
        start_time + double(i%points_per_channel)/(points_per_channel - 1)
        *(end_time - start_time);
    }
  }

  return 0;
}