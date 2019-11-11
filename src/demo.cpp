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



void nv_camera_stream(){
    gstCamera* camera = getCamera();
    bool user_quit = false;
    int imgWidth = camera->GetWidth();
    int imgHeight = camera->GetHeight();
    double fps = 10.0;
    clock_t clk;
    int num_dets = 0;
    std::vector<std::string> label_encodings;

    //create networks
    mtcnn finder(imgHeight, imgWidth);            //detection network
    face_embedder embedder;                     //recognition network, generate face embeddings
    face_classifier classifier(&embedder);
    glDisplay* display = getDisplay();
    classifier.get_label_encoding(&label_encodings);
    
    //malloc shared memory for images for access with cpu and gpu without copying data
    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, imgWidth*imgHeight*3*sizeof(uchar) );
    uchar* cropped_buffer_gpu[2] = {NULL,NULL};
    uchar* cropped_buffer_cpu[2] = {NULL,NULL};
    cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
    cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) );
  
    while(!user_quit){
        
        clk = clock();              //fps clock
		float* imgOrigin = NULL;    //camera image  
        // the 2nd arg 1000 defines timeout, true is for the "zeroCopy" means the image will be at shared memory        
        if( !camera->CaptureRGBA(&imgOrigin, 1000, true))
			printf("failed to capture RGBA image from camera\n");
        
        //since the captured image is located at shared memory, we easily can access it from cpu 
        // - we even can define a cv::Mat for it without copying data. We use the Mat only to draw the detections onto the Image
        cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, imgOrigin);

        //the mtcnn pipeline is based on GpuMat 8bit values 3 channels - so we remove the A-channel and size vals down
        cudaRGBA32ToBGR8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );      //Transform the memory layout 
        //cudaRGBA32ToRGB8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );    //Transform the memory layout  TODO: test which one is better
        cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);                 // Image as opencv cuda format

        //pass image to detection Network MTCNN and get face detections
        std::vector<struct Bbox> detections;
        finder.findFace(imgRGB_gpu, &detections);

        std::vector<cv::Rect> rects;
        std::vector<float*> keypoints;
        std::vector<double> face_labels;
        std::vector<matrix<rgb_pixel>> faces;

        num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints);        
        crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
        
        //for testing: show cropped and aligned faces in a separate window for visual check if alignment works well
       /* for(int i=0; i<num_dets; i++){
            image_window my_win(faces[i], "win");
            my_win.wait_until_closed();
        }*/
        
        //pass the detected faces to the recognition network
        //feed the face embeddings into a SVM
        if(num_dets > 0){
            std::vector<matrix<float,0,1>> face_embeddings;
            embedder.embeddings(&faces, &face_embeddings);
            // layout
            // rect 1       rect 2     rect 3    ...
            // face 1       face 2     face 3    ...
            // label 1      label 2    label 3   ...
            // embedding1   emb 2      emb 3     ...
            
            // SVM prediction
            classifier.prediction(&face_embeddings, &face_labels);
            //for(int i = 0; i<face_labels.size(); i++){
            //    cout << "detected face: " << face_labels[i] << " - " << label_encodings[face_labels[i]] << endl;
            //}
            draw_detections(origin_cpu, &rects, &face_labels, &label_encodings);
            
            // TODO: handle detections - reaction
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






void ttt(){
    
    face_embedder embedder;

    face_classifier fc(&embedder);



}

int main()
{
    //test();
    //ttt();
    nv_camera_stream();
    //nv_image_test("4.jpg");

    return 0;
}
