#include "network.h"
#include "mtcnn.h"
#include "kernels.h"
#include <time.h>
#include <math.h>
#include <opencv2/cudawarping.hpp>

#include "includes/gstCamera.h"
#include "includes/glDisplay.h"
#include "includes/loadImage.h"
#include "includes/cudaRGB.h"
#include "includes/cudaMappedMemory.h"
#include "includes/cudaOverlay.h"

#include "recognize.h"

#define PI 3.14159265


gstCamera* getCamera(){
    gstCamera* camera = gstCamera::Create(gstCamera::DefaultWidth, gstCamera::DefaultHeight, NULL);
	if( !camera ){
		printf("\nfailed to initialize camera device\n");
	}else{
        printf("\nsuccessfully initialized camera device\n");
        printf("    width:  %u\n", camera->GetWidth());
        printf("   height:  %u\n", camera->GetHeight());
        printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
            //start streaming
	    if( !camera->Open() ){
            printf("failed to open camera for streaming\n");
	    }else{
            printf("camera open for streaming\n");
        }
    }
    return camera;
}


glDisplay* getDisplay(){
    glDisplay* display = glDisplay::Create();
	if( !display ) 
		printf("failed to create openGL display\n");
    return display;
}


/*

void nv_image_test(const char* imgFilename){
	glDisplay* display = getDisplay();
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
    uchar* rgb = NULL;
    
    int imgWidth = 0;
    int imgHeight = 0;
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
	}
    CHECK(cudaMalloc(&rgb, imgWidth*imgHeight*3*sizeof(uchar)));
    cudaRGBA32ToBGR8( (float4*)imgCUDA, (uchar3*)rgb, imgWidth, imgHeight );
    cv::cuda::GpuMat imageRGB(imgHeight, imgWidth, CV_8UC3, rgb);
    
    //test: load image to cpu and show it
    cv::Mat cpuImage;
    imageRGB.download(cpuImage); 
    cv::imshow("result", cpuImage);
    cv::waitKey(0);
    

    mtcnn find(imgHeight, imgWidth);
    clock_t start;

    for(int i=0;i<10;i++){
        start = clock();
        find.findFace(imageRGB);
        start = clock() -start;
        cout<<"time is  "<<1000*(double)start/CLOCKS_PER_SEC<<endl;
    }
    SAFE_DELETE(display);
    CHECK(cudaFree(rgb));
}*/


double get_face_rotation(float left_x, float left_y, float right_x, float right_y){
    float dy = left_y - right_y; 
    float dx = right_x - left_x;
    double rotation = atan(dy/dx);
    return -rotation;
}

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

void get_face_shift(float* keypoints, double *x_shift, double *y_shift){
    
    if(*(keypoints + 2) - *(keypoints) >= 25){          //if nose is closer to the right eye
        *x_shift = 50 - (double) *(keypoints);          //left eye alignment to coordinates (50|50)
        *y_shift = 50 - (double) *(keypoints + 5);
    }else{                                              // nose is closer to the left eye
        *x_shift = 100 - (double) *(keypoints + 1);     //right eye alignment to coordinates (100|50)
        *y_shift = 50 - (double) *(keypoints + 6);
    }
    //printf("shift: %f | %f\n", *x_shift, *y_shift);

    //TODO: detect bad face images and discard them - like a keypoint is out of the rectangle or something like that
}


void crop_and_align_faces(cv::cuda::GpuMat &gpuImage,                     //raw image
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
        rotation_rad = get_face_rotation(*(keypoints[0][cnt]), *(keypoints[0][cnt]+5) ,*(keypoints[0][cnt]+1) ,*(keypoints[0][cnt]+6));
        rotation_deg = rotation_rad  * 180 / PI;
        
        //apply the resize operation and the rotation to the keypoints to have them available for alignment
        get_new_keypoints(-rotation_rad, &(*it), keypoints[0][cnt]);

        //after resize and rotation we want to shift the face to align the keypoints as it's required by the recognition-CNN
        //optimal for frontal faces: right eye at (100|50), left eye at(50|50), left mouth at(50|100) right mouth at (100|100), nose (75|75)
        //to make it as easy as possible we align one eye, depending on the face pose. 
        //The general face dimension should be quite OK after resize
        get_face_shift(keypoints[0][cnt], &x_shift, &y_shift);

        //apply the alignment to the face - the calculated rotation and the shift of the face 
        cv::cuda::rotate(crop_gpu_buffer, crop_gpu, cv::Size(150, 150), rotation_deg, x_shift, y_shift);  

        //for testing: draw the new keypoints to the cropped face to make sure we calculated right coordinates 
        for(int num=0;num<5;num++){
            *(keypoints[0][cnt]+num) = (int)*(keypoints[0][cnt]+num) + x_shift;
            *(keypoints[0][cnt]+num+5) = (int)*(keypoints[0][cnt]+num+5) + y_shift;
            cv::circle(crop_cpu,cv::Point((int)*(keypoints[0][cnt]+num) , (int)*(keypoints[0][cnt]+num+5)),3,cv::Scalar(255,0,0), -1);     
        }
        cudaDeviceSynchronize();  

        //generate network inputs from cv-format to the required dlib-format
        dlib::cv_image<rgb_pixel> image(crop_cpu);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, image);
        //store the aligned face to a CNN-input vector 
        cropped_faces->push_back(matrix);
    }
}


//transform MTCNN detection to cv rectangles and keypoints
//draw the detections to the original image
int get_detections( cv::Mat &origin_cpu, std::vector<struct Bbox> *detections, 
                    std::vector<cv::Rect> *rects, std::vector<float*> *keypoints, bool draw = false){
    int cnt = 0;

    for(std::vector<struct Bbox>::iterator it=(*detections).begin(); it!=(*detections).end();it++){
        if((*it).exist){
            //extract boxes and keypoints
            cnt ++;
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            (*rects).push_back(temp);          
            keypoints->push_back(it->ppoint);

            //draw box and keypoints to the original image
            if(draw == true){
                cv::rectangle(origin_cpu, temp, cv::Scalar(0,0,255), 2,8,0);
                for(int num=0;num<5;num++)cv::circle(origin_cpu,cv::Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,cv::Scalar(255,0,0), -1); 
            }
        }
    }
    return cnt;
}

                     

void nv_camera_stream(){
    gstCamera* camera = getCamera();
    bool user_quit = false;
    int imgWidth = camera->GetWidth();
    int imgHeight = camera->GetHeight();
    double fps = 10.0;
    clock_t clk;
    int num_dets = 0;

    //create networks
    mtcnn find(imgHeight, imgWidth);    //detection network
    recognize rec;                      //recognition network, generate face embeddings
    rec.init();                         //run a first recognition to set the network up
    
    glDisplay* display = getDisplay();
    
    //malloc shared memory for access with cpu and gpu without copying data
    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, imgWidth*imgHeight*3*sizeof(uchar) );
    uchar* cropped_buffer_gpu[2] = {NULL,NULL};
    uchar* cropped_buffer_cpu[2] = {NULL,NULL};
    cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
    cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) );
  
    while(!user_quit){
        //fps clock
        clk = clock();
        //camera image
		float* imgOrigin = NULL;   
        // the 2nd arg 1000 defines timeout, true is for the "zeroCopy"-option which means the image will be at shared memory        
        if( !camera->CaptureRGBA(&imgOrigin, 1000, true))
			printf("failed to capture RGBA image from camera\n");
        
        //since the captured image is located at shared mem, we can easily access it from cpu 
        // - we even can define a Mat for it without copying data. We use the Mat only to draw the detections onto the Image
        cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, imgOrigin);

        //the mtcnn pipeline is based on GpuMat 8bit values 3 channels - so we remove the A-channel and size vals down
        cudaRGBA32ToBGR8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );    //correct colors for cv (and for mtcnn?)
        //cudaRGBA32ToRGB8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );  //MTCNN better Performance? TODO: test it
        cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);
        //cv::Mat imgRGB_cpu(imgHeight, imgWidth, CV_8UC3, rgb_cpu);

        //pass image to detection Network MTCNN and get face detections
        std::vector<struct Bbox> detections;
        find.findFace(imgRGB_gpu, &detections);

        //create cv rectangles and keypoints from the detections and draw them to the origin image (if arg 5 is true)
        std::vector<cv::Rect> rects;
        std::vector<float*> keypoints;
        num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints, true);

        //crop and align the faces -> generate inputs for the recognition CNN
        std::vector<matrix<rgb_pixel>> faces;
        crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
        
        //for testing: show cropped and aligned faces in a separate window for visual check if alignment works well
        //for(int i=0; i<num_dets; i++){
        //    image_window my_win(faces[i], "win");
        //    my_win.wait_until_closed();
        //}

        //pass the detected faces to the recognition network
        //TODO: feed the face embeddings into a SVM
        if(num_dets > 0){
            std::vector<matrix<float,0,1>> face_descriptors;
            rec.embeddings(&faces, &face_descriptors);
            printf("%d detections!\n",num_dets);
            //face_descriptors -> SVM
            //prediction!
        }

        //Render captured image, print the FPS to the windowbar
        if( display != NULL ){
            display->RenderOnce(imgOrigin, imgWidth, imgHeight);
            char str[256];
            fps = (0.95 * fps) + (0.05 * (1 / ((double)(clock()-clk)/CLOCKS_PER_SEC)));
            sprintf(str, "TensorRT  %.0f FPS", fps);
            display->SetTitle(str);
            //check if user quit
            if( display->IsClosed() )
				user_quit = true;
        }
    }   
    SAFE_DELETE(camera);
    SAFE_DELETE(display);
    CHECK(cudaFreeHost(rgb_cpu));
    CHECK(cudaFreeHost(cropped_buffer_cpu[0]));
    CHECK(cudaFreeHost(cropped_buffer_cpu[1]));
}


/*
void recog(){
    recognize rec;   
    rec.tell();

    float *cpu = NULL;
    float *gpu = NULL;
    size_t size = 150 * 150 * 3 * sizeof(float);
    //cudaAllocMapped( (void**) &cpu, (void**) &gpu, size );
    
    
    //resizable_tensor im;
   // set_tensor(&im, adress);
    //tensorius.set_size(std::distance(ibegin,iend), 3, NR, NC);
    


    matrix<rgb_pixel> img;
    load_image(img, "res.jpg");
    image_window my_win(img, "win");
    my_win.wait_until_closed();
    rec.embeddings(&img);

    std::vector<matrix<rgb_pixel>> im;
    im.push_back(img);
    rec.embeddings(&im);


    //rec.tell();
    
    clock_t clk;
    for(int i=0; i<100; i++){
        clk = clock();
        rec.embeddings(&im);
        cout << "\nTime : " << 1000*(double)(clock()-clk)/CLOCKS_PER_SEC << endl;
    }

}*/

void test(){

    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, 1000*3*sizeof(uchar) );
    


    //CHECK(cudaFree(rgb_gpu));
    CHECK(cudaFreeHost(rgb_cpu));
}


int main()
{

    //test();
    //recog();
    nv_camera_stream();
    //nv_image_test("4.jpg");
    //cv_image_test("4.jpg");

    return 0;
}
