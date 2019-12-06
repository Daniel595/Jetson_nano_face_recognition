//Author : Daniel595

#ifndef FACE_EMBEDDER_H
#define FACE_EMBEDDER_H
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
//#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
//#include <dlib/image_processing/frontal_face_detector.h>
#include "network.h"
#include "dlib/opencv/cv_image.h"

// Header of embeddings.cpp and classification.cpp

using namespace dlib;
using namespace std;

class face_embedder
{
public:

    face_embedder(void);
    ~face_embedder(void);

    //embeddings.cpp
    void embeddings(std::vector<matrix<rgb_pixel>> *face_chips, std::vector<matrix<float,0,1>> *face_embeddings);
    void embedding(matrix<rgb_pixel> *face_chip, matrix<float,0,1> *face_embedding);
    void tell();
    void init();

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;
    
    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET> 
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                alevel0<
                                alevel1<
                                alevel2<
                                alevel3<
                                alevel4<
                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                input_rgb_image_sized<150>
                                >>>>>>>>>>>>;



private:
    //embeddings.cpp
    anet_type net;
    //shape_predictor sp; //unused since we so the alignment manual in alignment.h
};

#endif
