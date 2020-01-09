//Author : Daniel595

#include "alignment.h"

//local prototypes


double get_face_rotation(float left_x, float left_y, float right_x, float right_y);
void get_new_keypoints(double rotation, cv::Rect *face, float* keypoints);
void get_face_shift(float* keypoints, double *x_shift, double *y_shift);


//----------------------------------------------------------------------------------------------------

//defines

// get rotation of detected face
//  - find out how much the face is rotated (the eyes should make a horizontal line)
//  - returns rotation
double get_face_rotation(float left_x, float left_y, float right_x, float right_y){
    float dy = left_y - right_y; 
    float dx = right_x - left_x;
    double rotation = atan(dy/dx);
    return -rotation;
}

// perform a rotation of the 5 keypoints to make them fit to the rotated face
void get_new_keypoints(double rotation, cv::Rect *face, float* keypoints){
    const int FACE_SIZE = 150;
    //the point where the rotation is around - in this case its (0|0)(top left corner), given by opencv
    int x0 = 0;
    int y0 = 0;
    //rectangles border offset - from the rectangle edge to the picture edge
    int xmin = face->x;
    int ymin = face->y;
    //rectangle size
    int width = face->width;
    int height = face->height;   

    //calc the new coordinates of the keypoints   
    for(int i =0; i<5; i++){
        
        // apply the resize to 150*150 to the keypoints - new dimension * relation of keypoints 
        float x = FACE_SIZE * ( (*(keypoints + i) - (xmin)) / width );
        float y = FACE_SIZE * ( (*(keypoints + i+5) - (ymin)) / height );
        
        // apply the rotation around the point (x0|y0) to the keypoints
        *(keypoints + i) = (x - x0)*cos(rotation) - (y-y0)*sin(rotation)  + x0;         //new X coordinate
        *(keypoints + i + 5) =  (x - x0)*sin(rotation) + (y-y0)*cos(rotation)  + y0;    //new Y coordinate
        //printf("%d: (%g | %g) \n",i,*(keypoints + i), *(keypoints + i+5));  
    }
}


// check if the face needs to be shifted to x and y (for embedding network alignment) 
void get_face_shift(float* keypoints, double *x_shift, double *y_shift){  

    // side view to the face from right ?
    //  nose_x          >   left_eye_x
    if(*(keypoints + 2) < *(keypoints) ){
        // assign right eye to middle
        *x_shift = 75 - (double) *(keypoints + 1);     // right eye alignment to coordinates (75|50)
        *y_shift = 50 - (double) *(keypoints + 6);

    // side view to the face from left ? 
    //          nose_x       <      right_eye_x
    }else if(*(keypoints + 2) > *(keypoints+1) ){
        // assign left eye to middle
        *x_shift = 75 - (double) *(keypoints);          // left eye alignment to coordinates (75|50)
        *y_shift = 50 - (double) *(keypoints + 5);
    }else{
        // nose closer to the right eye?
        //    nose_X        -   left_eye_x
        if(*(keypoints + 2) - *(keypoints) >= 25){          // if nose-x is closer to the right eye
            *x_shift = 50 - (double) *(keypoints);          // left eye alignment to coordinates (50|50)
            *y_shift = 50 - (double) *(keypoints + 5);
        // nose closer to the left eye?
        }else{                                              // nose is closer to the left eye
            *x_shift = 100 - (double) *(keypoints + 1);     // right eye alignment to coordinates (100|50)
            *y_shift = 50 - (double) *(keypoints + 6);
        } 
    }
    //printf("shift: %f | %f\n", *x_shift, *y_shift);
    //TODO: detect bad face images (?) and discard them
}


// Get a cropped an aligned face (prepared for face embedding network) from the MTCNN detection
//  - resize faces to 150 x 150
//  - ensure zero rotation of the face (eyes make a horizontal line)
//  - face aligned (for information look at comments in the following code) 
//  - store the aligned faces to the handed over vector for further use
void crop_and_align_faces(cv::cuda::GpuMat &gpuImage,           //raw image
                uchar *gpu_adress[2], uchar *cpu_adress[2],     //shared memory for cropped faces
                std::vector<cv::Rect> *rects,                   //rectangles with detected faces
                std::vector<matrix<rgb_pixel>> *cropped_faces,  //vector to store cropped faces
                std::vector<float*> *keypoints                  //face keypoints
                )
{
    cv::cuda::GpuMat crop_gpu(150, 150, CV_8UC3, gpu_adress[0]);
    cv::cuda::GpuMat crop_gpu_buffer(150, 150, CV_8UC3, gpu_adress[1]);
    cv::Mat crop_cpu(150, 150, CV_8UC3, cpu_adress[0]);
    //cv::Mat crop_cpu_2(150, 150, CV_8UC3, cpu_adress[1]);

    int cnt = 0;
    double x_shift = 0;
    double y_shift = 0;
    double rotation_rad = 0;
    double rotation_deg = 0;

    for(std::vector<cv::Rect>::iterator it=(*rects).begin(); it!=(*rects).end();it++){
        
        //cut out the face and size it to a 150*150 face_chip, which is needed by the recognition CNN
        cv::cuda::resize(gpuImage(*it), crop_gpu_buffer, cv::Size(150, 150), 0, 0, cv::INTER_LINEAR);
        
        //calculate the "rotation" of the face. Its done by the keypoints of the left and the right eye.
        //for an aligned face, both eyes should be on the same y-coords.
        rotation_rad = get_face_rotation( *(keypoints->at(cnt)), *(keypoints->at(cnt) + 5) ,*(keypoints->at(cnt) + 1) , *(keypoints->at(cnt) + 6) );
        rotation_deg = rotation_rad  * 180 / PI;
        
        //apply the resize operation and the rotation to the keypoints to have them available for alignment
        get_new_keypoints(-rotation_rad, &(*it), keypoints->at(cnt) );

        //after resize and rotation we want to shift the face to align the keypoints as it's required by the recognition-CNN
        //optimal for frontal faces: right eye at (100|50), left eye at(50|50), left mouth at(50|100) right mouth at (100|100), nose (75|75)
        //to make it as easy as possible we align one eye, depending on the face pose. 
        //The general face dimension should be quite OK after resize
        get_face_shift(keypoints->at(cnt) , &x_shift, &y_shift);

        // apply the alignment to the face - the calculated rotation and the shift of the face 
        cv::cuda::rotate(crop_gpu_buffer, crop_gpu, cv::Size(150, 150), rotation_deg, x_shift, y_shift);  
        // convert to RGB since its the format the Embedding CNN is trained for
        cv::cuda::cvtColor(crop_gpu, crop_gpu, cv::COLOR_BGR2RGB);  
        
        //for testing: draw the new keypoints to the cropped face to make sure we calculated right coordinates 
        /////Test
        //for(int num=0;num<5;num++){
        //    *(keypoints->at(cnt) + num) = (int)*(keypoints->at(cnt) + num) + x_shift;
        //    *(keypoints->at(cnt) + num + 5) = (int)*(keypoints->at(cnt) + num + 5) + y_shift;
        //    cv::circle(crop_cpu,cv::Point((int)*(keypoints->at(cnt) + num) , (int)*(keypoints->at(cnt) + num + 5)),3,cv::Scalar(255,0,0), -1);     
        //}

        cudaDeviceSynchronize();  

        //generate network inputs from cv-format to the required dlib-format
        dlib::cv_image<rgb_pixel> image(crop_cpu);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, image);
        //store the aligned face to a CNN-input vector 
        cropped_faces->push_back(matrix);
        cnt++;
    }

    // test - show first detected face
    //image_window my_win(cropped_faces->at(0), "face");
    //my_win.wait_until_closed();
}


// transform MTCNN detection 
//  - returns the number of detected faces in this image
//  - create cv-rectangles and keypoints for further use
//  - optional: draw the keypoints
int get_detections( cv::Mat &origin_cpu, std::vector<struct Bbox> *detections, 
                    std::vector<cv::Rect> *rects, std::vector<float*> *keypoints, bool draw){
    int cnt = 0;

    for(std::vector<struct Bbox>::iterator it=(*detections).begin(); it!=(*detections).end();it++){
        if((*it).exist){
            //extract boxes and keypoints
            cnt ++;
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            (*rects).push_back(temp);          
            keypoints->push_back(it->ppoint);
            if(draw){
                for(int num=0; num<5; num++){
                    cv::circle(origin_cpu,cv::Point((int)*(it->ppoint + num) , 
                        (int)*(it->ppoint + num + 5)),3,cv::Scalar(0,255,0,255), -2); 
                }
            }
        }
    }
    return cnt;
}



// Draw detections onto the current image
//  - draw the face bounding boxes
//  - draw the predicted labels at the top of the bounding box
void draw_detections(   cv::Mat &origin_cpu, 
                        std::vector<cv::Rect> *rects, 
                        std::vector<double> *labels,
                        std::vector<std::string> *label_encodings){
        
    for(int i = 0; i<rects->size(); i++){
        cv::Scalar bbox_color(0,255,0,255);
        // get label 
        string encoding = "";
        if(labels->at(i) >= 0){
            encoding =  label_encodings->at(labels->at(i));
        }else{
            encoding = "Unknown";
            bbox_color=cv::Scalar(255,0,0,255);
        }
        // draw bounding boxes around the face
        cv::rectangle(origin_cpu, rects->at(i), bbox_color, 2,8,0);

        // print label to the bounding box
        cv::putText(origin_cpu, encoding , cv::Point(rects->at(i).x,rects->at(i).y), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255,255,255,255), 3 ); // mat, text, coord, font, scale, bgr color, line thickness
        cv::putText(origin_cpu, encoding , cv::Point(rects->at(i).x,rects->at(i).y), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0,255), 1 );

    }                        
}

