#include "network.h"
#include "mtcnn.h"
#include "kernels.h"
#include <time.h>

#include "includes/gstCamera.h"
#include "includes/glDisplay.h"
#include "includes/loadImage.h"
#include "includes/cudaRGB.h"

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
    
    /*//test: load image to cpu and show it
    cv::Mat cpuImage;
    imageRGB.download(cpuImage); 
    cv::imshow("result", cpuImage);
    cv::waitKey(0);
    */

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
}





void nv_camera_stream(){
    gstCamera* camera = getCamera();
    glDisplay* display = getDisplay();
    bool user_quit = false;
    int imgWidth = camera->GetWidth();
    int imgHeight = camera->GetHeight();
    double fps = 10.0;
    
    mtcnn find(imgHeight, imgWidth);    //create network
    uchar* rgb = NULL;
    CHECK(cudaMalloc(&rgb, imgWidth*imgHeight*3*sizeof(uchar)));
    
    clock_t clk;
    while(!user_quit){
        clk = clock();
		float* imgRGBA = NULL;          //capture image
        if( !camera->CaptureRGBA(&imgRGBA, 1000))
			printf("failed to capture RGBA image from camera\n");

        //image to cv2 GpuMat
        cudaRGBA32ToBGR8( (float4*)imgRGBA, (uchar3*)rgb, imgWidth, imgHeight );
        cv::cuda::GpuMat imgRGB(imgHeight, imgWidth, CV_8UC3, rgb);
        //image to Network
        find.findFace(imgRGB);
        
        //Render captured image (Todo: add detection overlays(CUDA))
        if( display != NULL ){
            display->RenderOnce(imgRGBA, imgWidth, imgHeight);
            char str[256];
            fps = 1 / ((double)(clock()-clk)/CLOCKS_PER_SEC);
            sprintf(str, "TensorRT  %.0f FPS", fps);
            display->SetTitle(str);
            //check if user quit
            if( display->IsClosed() )
				user_quit = true;
        }
    }   
    SAFE_DELETE(camera);
    SAFE_DELETE(display);
    CHECK(cudaFree(rgb));
}







void cv_image_test(string image_path)
{
    cudaDeviceProp prop;
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    if (!prop.deviceOverlap)
    {
        printf("No device will handle overlaps. so no speed up from stream.\n");
        return ;
    }
    //test using images
    cv::Mat image = cv::imread(image_path);
    cv::cuda::GpuMat Gpuimage(image);
    cv::cuda::resize(Gpuimage,Gpuimage,cv::Size(1280,image.rows*1280/image.cols));
    mtcnn find(Gpuimage.rows, Gpuimage.cols);
    clock_t start;
    for(int i=0;i<100;i++){
        start = clock();
        find.findFace(Gpuimage);
        start = clock() -start;
        cout<<"time is  "<<1000*(double)start/CLOCKS_PER_SEC<<endl;
        cv::imshow("result", image);
    }
    cv::imwrite("result.jpg",image);
    cv::waitKey(0);
    image.release();
}



int main()
{

    nv_camera_stream();
    //nv_image_test("4.jpg");
    //cv_image_test("4.jpg");

    return 0;
}
