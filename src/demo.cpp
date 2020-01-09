#include "test.h"   // contains all required includes 



// create high performace jetson camera - loads the image directly to gpu memory or shared memory
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


// perform face recognition with Raspberry Pi camera
int camera_face_recognition(){
    
    // -------------- Initialization -------------------
    
    face_embedder embedder;                         // deserialize recognition network 
    face_classifier classifier(&embedder);          // train OR deserialize classification SVM's 
    if(classifier.need_restart() == 1) return 1;    // small workaround - if svms were trained theres some kind of memory problem when generate mtcnn

    // create camera
    gstCamera* camera = getCamera();                // create jetson camera - PiCamera. USB-Cam needs different operations in Loop!! not implemented!
    bool user_quit = false;
    int imgWidth = camera->GetWidth();
    int imgHeight = camera->GetHeight();

    mtcnn finder(imgHeight, imgWidth);              // build OR deserialize TensorRT detection network
    glDisplay* display = test::getDisplay();              // create jetson display
    
    // malloc shared memory for images for access with cpu and gpu without copying data
    // cudaAllocMapped is used from jetson-inference
    uchar* rgb_gpu = NULL;
    uchar* rgb_cpu = NULL;
    cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, imgWidth*imgHeight*3*sizeof(uchar) );
    uchar* cropped_buffer_gpu[2] = {NULL,NULL};
    uchar* cropped_buffer_cpu[2] = {NULL,NULL};
    cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
    cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) );
    
    // calculate fps
    double fps = 0.0;
    clock_t clk;
    
    // Detection vars
    int num_dets = 0;
    std::vector<std::string> label_encodings;       // vector for the real names of the classes/persons

    // get the possible class names
    classifier.get_label_encoding(&label_encodings);
    
    
    // ------------------ "Detection" Loop -----------------------

    while(!user_quit){

        clk = clock();              // fps clock
		float* imgOrigin = NULL;    // camera image  
        // the 2nd arg 1000 defines timeout, true is for the "zeroCopy" param what means the image will be stored to shared memory          
        if( !camera->CaptureRGBA(&imgOrigin, 1000, true))                                   
			printf("failed to capture RGBA image from camera\n");
        
        //since the captured image is located at shared memory, we also can access it from cpu 
        // here I define a cv::Mat for it to draw onto the image from CPU without copying data -- TODO: draw from CUDA
        cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, imgOrigin);

        // the mtcnn pipeline is based on GpuMat 8bit values 3 channels while the captured image is RGBA32
        // i use a kernel from jetson-inference to remove the A-channel and float to uint8
        cudaRGBA32ToBGR8( (float4*)imgOrigin, (uchar3*)rgb_gpu, imgWidth, imgHeight );      
        
        // create GpuMat form the same image thanks to shared memory
        cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);                

        // pass the image to the MTCNN and get face detections
        std::vector<struct Bbox> detections;
        finder.findFace(imgRGB_gpu, &detections);

        // check if faces were detected, get face locations, bounding boxes and keypoints
        std::vector<cv::Rect> rects;
        std::vector<float*> keypoints;
        num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints);            
        
        // if faces detected
        if(num_dets > 0){
            // crop and align the faces. Get faces to format for "dlib_face_recognition_model" to create embeddings
            std::vector<matrix<rgb_pixel>> faces;                                   
            crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
            
            // generate face embeddings from the cropped faces and store them in a vector
            std::vector<matrix<float,0,1>> face_embeddings;
            embedder.embeddings(&faces, &face_embeddings);                        
           
            // feed the embeddings to the pretrained SVM's. Store the predicted labels in a vector
            std::vector<double> face_labels;
            classifier.prediction(&face_embeddings, &face_labels);                 

            // draw bounding boxes and labels to the original image 
            draw_detections(origin_cpu, &rects, &face_labels, &label_encodings);    
        }

        //Render captured image
        if( display != NULL ){
            display->RenderOnce(imgOrigin, imgWidth, imgHeight);   // display image directly from gpumem/sharedmem
            char str[256];
            sprintf(str, "TensorRT  %.0f FPS", fps);               // print the FPS to the bar
            display->SetTitle(str);
            if( display->IsClosed() )                               //check if user quit
				user_quit = true;
        }
        // smooth FPS to make it readable
        fps = (0.90 * fps) + (0.1 * (1 / ((double)(clock()-clk)/CLOCKS_PER_SEC)));      
    }   

    SAFE_DELETE(camera);
    SAFE_DELETE(display);
    CHECK(cudaFreeHost(rgb_cpu));
    CHECK(cudaFreeHost(cropped_buffer_cpu[0]));
    CHECK(cudaFreeHost(cropped_buffer_cpu[1]));
    return 0;
}




int main()
{
    int state = 0;

    // predict camera input
    state = camera_face_recognition();
    
    // test prediction at a set of test images
    //state = test::test_prediction_images();
    
    // test prediction fps on a single image
    //state = test::test_prediction_fps_image("pictures/fps/bbt_4.png", "pictures/fps/result_4.png");
    
    // test MTCNN detection fps on single image
    //test::test_detection_fps_image("pictures/fps/bbt_4.png", "pictures/fps/result_4.png");

    if(state == 1) cout << "Restart is required! Please type ./main again." << endl;
    
    return 0;
}
