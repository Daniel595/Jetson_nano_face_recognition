#include <math.h>
#include <opencv2/cudawarping.hpp>
#include "network.h"
#include "mtcnn.h"

#define PI 3.14159265

double get_face_rotation(float left_x, float left_y, float right_x, float right_y);

void get_new_keypoints(double rotation, cv::Rect *face, float* keypoints);

void get_face_shift(float* keypoints, double *x_shift, double *y_shift);

void crop_and_align_faces(cv::cuda::GpuMat &gpuImage,uchar *gpu_adress[2], uchar *cpu_adress[2],     
                std::vector<cv::Rect> *rects, std::vector<matrix<rgb_pixel>> *cropped_faces,  
                std::vector<float*> *keypoints);

int get_detections( cv::Mat &origin_cpu, std::vector<struct Bbox> *detections, 
                    std::vector<cv::Rect> *rects, std::vector<float*> *keypoints, bool draw = false);