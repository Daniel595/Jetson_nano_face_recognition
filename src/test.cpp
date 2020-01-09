#include "test.h"

namespace test{




    // create high performance jetson display - show image directly from gpumem/sharedmem
    glDisplay* getDisplay(){
        glDisplay* display = glDisplay::Create();
        if( !display ) 
            printf("failed to create openGL display\n");
        return display;
    }






    // perform face recognition on test images
    int test_prediction_images(){
        
        // required format
        int wid = 1280;         // input size width
        int height = 720;       // input size height
        int channel = 3;        // all images rgb
        int face_size = 150;    // cropped faces are always square 150x150 px (required for dlib face embedding cnn)

        using namespace boost::filesystem;
        path p("faces/bbt/testdata/test/");     // directory with bbt-testdata
        
        // get recognition network and classifier
        face_embedder embedder;                         
        face_classifier classifier(&embedder);          
        if(classifier.need_restart() == 1) return 1;  

        // get detection network
        mtcnn finder(height, wid);              
        int num_dets = 0;
        int num_images = 0;

        // get the possible class names
        std::vector<std::string> label_encodings;       
        classifier.get_label_encoding(&label_encodings);

        // GPU memory for image in MTCNN format
        uchar* rgb_gpu = NULL;
        uchar* rgb_cpu = NULL;
        cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, height*wid*channel*sizeof(uchar) );
        
        // GPU memory for cropped faces
        uchar* cropped_buffer_gpu[2] = {NULL,NULL};
        uchar* cropped_buffer_cpu[2] = {NULL,NULL};
        cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], face_size*face_size*channel*sizeof(uchar) );
        cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], face_size*face_size*channel*sizeof(uchar) );

        // gpu/cpu memory pointer to load image from disk    
        float* imgCPU    = NULL;
        float* imgCUDA   = NULL;
        int    imgWidth  = 0;
        int    imgHeight = 0;

        // read and process images

        try
        {
            if (exists(p))    
            {
                if (is_regular_file(p))
                    cout << p << "is a file" << endl;
                else if (is_directory(p))   
                {
                    // Iteration over all images in the test directory
                    recursive_directory_iterator dir(p), end;                 
                    while(dir != end){
                        
                        if(is_directory(dir->path())) {    
                            cout << "enter: " << dir->path().filename().string() << endl;
                        }
                        // Handle the images
                        else {      
                            
                            // load image from disk to gpu/cpu-mem. (shared mem. for access without copying)           
                            if( !loadImageRGBA(dir->path().string().c_str(), (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
                                printf("failed to load image '%s'\n", dir->path().filename().string().c_str());
                            
                            // check if size fits 
                            if((imgWidth != wid) || (imgHeight != height)){
                                cout << "image has wrong size!" << endl;
                            }else{
                                // create cv::Mat to draw detections from CPU (possible because of shared memory GPU/CPU)
                                cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, imgCPU);
                                // Convert image to format required by MTCNN
                                cudaRGBA32ToBGR8( (float4*)imgCUDA, (uchar3*)rgb_gpu, imgWidth, imgHeight );  
                                // create GpuMat which is required by MTCNN pipeline   
                                cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);                
                                std::vector<struct Bbox> detections;
                                // run MTCNN and get bounidng boxes of detected faces
                                finder.findFace(imgRGB_gpu, &detections);
                                std::vector<cv::Rect> rects;
                                std::vector<float*> keypoints;
                                num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints);             
                                if(num_dets > 0){
                                    // crop and align faces
                                    std::vector<matrix<rgb_pixel>> faces;                                  
                                    crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
                                    // get face embeddings - feature extraction
                                    std::vector<matrix<float,0,1>> face_embeddings;
                                    embedder.embeddings(&faces, &face_embeddings);        
                                    // do classification                 
                                    std::vector<double> face_labels;
                                    classifier.prediction(&face_embeddings, &face_labels);          
                                    // draw detection bbox and classification to the image        
                                    draw_detections(origin_cpu, &rects, &face_labels, &label_encodings);   
                                }        
                                CUDA(cudaDeviceSynchronize());    
                                
                                // save the predicted image              
                                string outputFilename = "faces/bbt/testdata/result/" + to_string(num_images) + ".png";
                                if( !saveImageRGBA(outputFilename.c_str(), (float4*)imgCPU, imgWidth, imgHeight, 255) )
                                    printf("failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, outputFilename.c_str());

                                num_images++;    
                            }    
                        }
                        CUDA(cudaDeviceSynchronize());
                        // free CUDA space to load next image
                        CUDA(cudaFreeHost(imgCPU));
                        ++dir;
                    }
                } else cout << p << " exists, but is neither a regular file nor a directory\n";    
            } else cout << p << " does not exist\n";                                              
            
                            
        }
        catch (const filesystem_error& ex)      
        {
            cout << ex.what() << '\n';
        }

        CHECK(cudaFreeHost(rgb_cpu));
        CHECK(cudaFreeHost(cropped_buffer_cpu[0]));
        CHECK(cudaFreeHost(cropped_buffer_cpu[1]));
        
        return 0;
    }







    int test_fps_image(const char* input_image, const char* output_image){
        face_embedder embedder;                         
        face_classifier classifier(&embedder);         
        if(classifier.need_restart() == 1) return 1;    
        mtcnn finder(720, 1280);              
        glDisplay* display = getDisplay();              
        uchar* rgb_gpu = NULL;
        uchar* rgb_cpu = NULL;
        cudaAllocMapped( (void**) &rgb_cpu, (void**) &rgb_gpu, 1280*720*3*sizeof(uchar) );
        uchar* cropped_buffer_gpu[2] = {NULL,NULL};
        uchar* cropped_buffer_cpu[2] = {NULL,NULL};
        cudaAllocMapped( (void**) &cropped_buffer_cpu[0], (void**) &cropped_buffer_gpu[0], 150*150*3*sizeof(uchar) );
        cudaAllocMapped( (void**) &cropped_buffer_cpu[1], (void**) &cropped_buffer_gpu[1], 150*150*3*sizeof(uchar) ); 
        float* imgCPU    = NULL;
        float* imgCUDA   = NULL;
        int    imgWidth  = 0;
        int    imgHeight = 0;

        double fps = 0.0;
        clock_t clk;
        int num_dets = 0;
        std::vector<std::string> label_encodings;      
        classifier.get_label_encoding(&label_encodings);      
        if( !loadImageRGBA(input_image, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
            printf("failed to load image \n");     
        bool user_quit = false;

        float* CPU_mem = NULL;
        float* GPU_mem = NULL;
        size_t size = 1280*720*4*sizeof(float);
        cudaAllocMapped( (void**) &CPU_mem, (void**) &GPU_mem,  size );

        while(!user_quit){
            if((imgWidth == 1280) && (imgHeight==720)){
                clk = clock();
                cudaMemcpy((void*)GPU_mem, (void*)imgCUDA, size, cudaMemcpyDeviceToDevice);

                cv::Mat origin_cpu(imgHeight, imgWidth, CV_32FC4, CPU_mem);
                cudaRGBA32ToBGR8( (float4*)GPU_mem, (uchar3*)rgb_gpu, imgWidth, imgHeight );      //Transform the memory layout 
                cv::cuda::GpuMat imgRGB_gpu(imgHeight, imgWidth, CV_8UC3, rgb_gpu);                 // Image as opencv cuda format
                std::vector<struct Bbox> detections;
                finder.findFace(imgRGB_gpu, &detections);
                std::vector<cv::Rect> rects;
                std::vector<float*> keypoints;
                num_dets = get_detections(origin_cpu, &detections, &rects, &keypoints);             // check for detections
                if(num_dets > 0){
                    std::vector<matrix<rgb_pixel>> faces;                                   // crop faces
                    crop_and_align_faces(imgRGB_gpu, cropped_buffer_gpu, cropped_buffer_cpu, &rects, &faces, &keypoints);
                    std::vector<matrix<float,0,1>> face_embeddings;
                    embedder.embeddings(&faces, &face_embeddings);                          // create embeddings
                    std::vector<double> face_labels;
                    classifier.prediction(&face_embeddings, &face_labels);                  // classification
                    draw_detections(origin_cpu, &rects, &face_labels, &label_encodings);    // draw detections
                }

                cv::putText(origin_cpu, to_string(fps) + " FPS", cv::Point(20,40), 
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255,0,0,255), 2 );

                if( display != NULL ){
                    display->RenderOnce(GPU_mem, imgWidth, imgHeight);                        // display image directly from gpumem/sharedmem
                    char str[256];
                    sprintf(str, "TensorRT  %.0f FPS", fps);                                    // print the FPS to the bar
                    display->SetTitle(str);
                    if( display->IsClosed() ){                                                   //check if user quit
                        user_quit = true;
                        saveImageRGBA(output_image, (float4*)CPU_mem, imgWidth, imgHeight, 255);
                    }
                }
                
                fps = (0.90 * fps) + (0.1 * (1 / ((double)(clock()-clk)/CLOCKS_PER_SEC)));      // get smooth FPS
            }else{
                cout << "wrong image size!" << endl;
                break;
            }
        }   

        SAFE_DELETE(display);
        CHECK(cudaFreeHost(rgb_cpu));
        CHECK(cudaFreeHost(cropped_buffer_cpu[0]));
        CHECK(cudaFreeHost(cropped_buffer_cpu[1]));
        CHECK(cudaFreeHost(CPU_mem));
        
        return 0;
    }




}   // namespace test end

