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

#include "recognize.h"
#include "alignment.h"




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
       /* for(int i=0; i<num_dets; i++){
            image_window my_win(faces[i], "win");
            my_win.wait_until_closed();
        }*/

        //pass the detected faces to the recognition network
        //TODO: feed the face embeddings into a SVM
        clock_t emb = clock();
        if(num_dets > 0){
            std::vector<matrix<float,0,1>> face_descriptors;
            rec.embeddings(&faces, &face_descriptors);
            cout << "embeddings: " << 1000* (clock() - emb)/CLOCKS_PER_SEC << "ms" << endl;
            //printf("%d detections!\n",num_dets);
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






void test(){
    
    using namespace boost::filesystem;
    
    path p ("train_data/database/");   // p reads clearer than argv[1] in the following code
    int num_labels = 0;
    path path_label = "";
    path path_file = "";
    std::ofstream o("labels.txt");

    std::vector<matrix<float,0,1>> vector_face_descriptors;
    std::vector<double> vector_labels;
    
    try
    {
        if (exists(p))    
        {
        if (is_regular_file(p))        
            cout << p << "is a file. \nStart at directory \"parent\", when the following structure is given: \"parent/label/image.jpg\" " << endl;
        else if (is_directory(p))
        {
            //iterate over all included directories and files
            recursive_directory_iterator dir(p), end; 
            while(dir != end){
                if(is_directory(dir->path())) {         // new label
                    path_label = dir->path().filename();
                    // write label to file
                    o << path_label.string() << endl;
                    num_labels ++;
                }else {                                 // file
                    path_file =  dir->path().filename();

                    // load file
                    // run detection
                    // extract faces
                    // create embeddings and push_back
                    // push_back label

                    cout << "label: " << num_labels - 1 << "   file: "<< path_file << endl;
                }
                ++dir;
            }
            o.close();
        }
        else
            cout << p << " exists, but is neither a regular file nor a directory\n";
        }
        else
        cout << p << " does not exist\n";
    }
    catch (const filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }

}



int main()
{
    //test();
    nv_camera_stream();
    //nv_image_test("4.jpg");

    return 0;
}
