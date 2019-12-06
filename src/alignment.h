//Author : Daniel595

#include <math.h>
#include <opencv2/cudawarping.hpp>
#include "network.h"
#include "mtcnn.h"
#include "face_embedder.h"

#define PI 3.14159265



void crop_and_align_faces(  cv::cuda::GpuMat &gpuImage,
                            uchar *gpu_adress[2], 
                            uchar *cpu_adress[2],     
                            std::vector<cv::Rect> *rects, 
                            std::vector<matrix<rgb_pixel>> *cropped_faces,  
                            std::vector<float*> *keypoints);




int get_detections( cv::Mat &origin_cpu, 
                    std::vector<struct Bbox> *detections, 
                    std::vector<cv::Rect> *rects, 
                    std::vector<float*> *keypoints,
                    bool draw=true);



void draw_detections(   cv::Mat &origin_cpu, 
                        std::vector<cv::Rect> *rects, 
                        std::vector<double> *labels,
                        std::vector<std::string> *label_encodings);