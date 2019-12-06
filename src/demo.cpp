#include "mtcnn.h"
#include "kernels.h"
#include <time.h>
#include <boost/filesystem.hpp>
#include <fstream>

#include "includes/gstCamera.h"
#include "includes/glDisplay.h"
#include "includes/loadImage.h"
#include "includes/cudaRGB.h"
#include "includes/cudaMappedMemory.h"

#include "face_embedder.h"
#include "face_classifier.h"
#include "alignment.h"

#include <dlib/svm_threaded.h>
#include <dlib/svm.h>
#include <vector>



// create high performace jetson camera - loads the image directly to gpumem/sharedmem
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

// create high performance jetson display - show image directly from gpumem/sharedmem
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
    
    //malloc shared memory for access with cpu and gpu without copying data
    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, imgWidth*imgHeight*3*sizeof(uchar) );
    uchar* cropped_buffer_gpu[2] = {NULL,NULL};
    uchar* cropped_buffer_cpu[2] = {NULL,NULL};
    cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
    cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) );

    cudaRGBA32ToBGR8( (float4*)imgCUDA, (uchar3*)rgb, imgWidth, imgHeight );
    cv::cuda::GpuMat imageRGB(imgHeight, imgWidth, CV_8UC3, rgb);
    
    //find faces
    mtcnn find(imgHeight, imgWidth);
    
    //create cv rectangles and keypoints from the detections and draw them to the origin image (if arg 5 is true)
    std::vector<cv::Rect> rects;
    std::vector<float*> keypoints;
    num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints, false);

    //crop and align the faces -> generate inputs for the recognition CNN
    std::vector<matrix<rgb_pixel>> faces;
    crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);

    for(int i=0;i<10;i++){
        start = clock();
        find.findFace(imageRGB);
    }


    SAFE_DELETE(display);
    CHECK(cudaFree(rgb));
}
*/



void run(){
    
    // -------------- Initialization -------------------

    // Camera stuff
    gstCamera* camera = getCamera();                // create jetson camera - PiCamera. USB-Cam needs different operations in Loop!! not implemented!
    bool user_quit = false;
    int imgWidth = camera->GetWidth();
    int imgHeight = camera->GetHeight();
    
    // FPS stuff
    double fps = 10.0;
    clock_t clk;
    
    // Detection vars
    int num_dets = 0;
    std::vector<std::string> label_encodings;       // vector for the real names of the classes/persons

    // create networks
    mtcnn finder(imgHeight, imgWidth);              // build OR deserialize TensorRT detection network
    face_embedder embedder;                         // deserialize recognition network 
    face_classifier classifier(&embedder);          // train OR deserialize classification SVM's 
    glDisplay* display = getDisplay();              // create jetson display
    
    // malloc shared memory for images for access with cpu and gpu without copying data
    // cudaAllocMapped is used from jetson-inference
    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, imgWidth*imgHeight*3*sizeof(uchar) );
    uchar* cropped_buffer_gpu[2] = {NULL,NULL};
    uchar* cropped_buffer_cpu[2] = {NULL,NULL};
    cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
    cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) );
    
    // get the possible class names
    classifier.get_label_encoding(&label_encodings);
    

    // ------------------ "Detection" Loop -----------------------

    while(!user_quit){

        clk = clock();              // fps clock
		float* imgOrigin = NULL;    // camera image  
        // the 2nd arg 1000 defines timeout, true is for the "zeroCopy" means the image will be located at shared memory what is pretty nice         
        if( !camera->CaptureRGBA(&imgOrigin, 1000, true))                                   // store image directly into shared memory ! 
			printf("failed to capture RGBA image from camera\n");
        
        //since the captured image is located at shared memory, we also can access it from cpu 
        // - we even can define a cv::Mat (cpu image format) for it without copying data. 
        // We use the Mat/CPU only to draw the detections onto the image - TODO: draw detections by CUDA
        cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, imgOrigin);

        // the mtcnn pipeline is based on GpuMat 8bit values 3 channels - so we remove the A-channel and size vals down
        // we've got a kernel from jetson-inference for that job, thank you guys
        cudaRGBA32ToBGR8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );      //Transform the memory layout 
        //cudaRGBA32ToRGB8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );    //Transform the memory layout  TODO: test which one is better
        
        // make a opencv gpumat header for the new image
        cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);                 // Image as opencv cuda format

        // pass the image to the MTCNN and get face detections
        std::vector<struct Bbox> detections;
        finder.findFace(imgRGB_gpu, &detections);

        // check if faces were detected, get face locations, bounding boxes and keypoints
        std::vector<cv::Rect> rects;
        std::vector<float*> keypoints;
        num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints);             // check for detections
        
        // if faces detected
        if(num_dets > 0){
            
            // crop and align the faces. Get faces to format for "dlib_face_recognition_model" to create embeddings
            std::vector<matrix<rgb_pixel>> faces;                                   // crop faces
            crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
            
            // generate face embeddings out of the cropped faces and store them in a vector
            std::vector<matrix<float,0,1>> face_embeddings;
            embedder.embeddings(&faces, &face_embeddings);                          // create embeddings
           
            // feed the embeddings to the pretrained SVM's. Store the predictions in a vector
            std::vector<double> face_labels;
            classifier.prediction(&face_embeddings, &face_labels);                  // classification

            // draw bounding boxes and labels to the original image (by CPU, TODO: draw from CUDA)
            draw_detections(origin_cpu, &rects, &face_labels, &label_encodings);    // draw detections
        }

        //Render captured image
        if( display != NULL ){
            display->RenderOnce(imgOrigin, imgWidth, imgHeight);                        // display image directly from gpumem/sharedmem
            char str[256];
            fps = (0.95 * fps) + (0.05 * (1 / ((double)(clock()-clk)/CLOCKS_PER_SEC))); // smooth FPS
            sprintf(str, "TensorRT  %.0f FPS", fps);                                    // print the FPS to the bar
            display->SetTitle(str);
            
            if( display->IsClosed() )                                                   //check if user quit
				user_quit = true;
        }
    }   

    SAFE_DELETE(camera);
    SAFE_DELETE(display);
    CHECK(cudaFreeHost(rgb_cpu));
    CHECK(cudaFreeHost(cropped_buffer_cpu[0]));
    CHECK(cudaFreeHost(cropped_buffer_cpu[1]));
}





int main()
{
    run();
    return 0;
}
