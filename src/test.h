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

namespace test{
    glDisplay* getDisplay();
    int test_fps_image(const char* input_image, const char* output_image);
    int test_prediction_images();
}