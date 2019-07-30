/*Base Header File */
#ifndef OPENCVLIB_INCL
#define OPENCVLIB_INCL

#ifdef _WIN32
#define OPENCVLIB_CALL _cdecl
#else
#define OPENCVLIB_CALL
#endif

#ifdef OPENCVLIB_EXPORTS
#define OPENCVLIB_API DECL_EXPORT
#else
#define OPENCVLIB_API DECL_IMPORT
#endif

#include "hqlplugins.hpp"
#include "eclrtl.hpp"

#include <stdio.h>
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/imgcodecs.hpp>

#include "./License_Plate_Files/DetectPlates/DetectPlates.h"
#include "./License_Plate_Files/PossibleChar/PossibleChar.h"
#include "./License_Plate_Files/PossiblePlate/PossiblePlate.h"
#include "./License_Plate_Files/Preprocess/Preprocess.h"
#include "./License_Plate_Files/DetectChars/DetectChars.h"
        

#ifdef OPENCVLIB_EXPORTS
extern "C"
{
    OPENCVLIB_API bool getECLPluginDefinition(ECLPluginDefinitionBlock *pb);
}
#endif

extern "C++"
{
    namespace OPENCVLib
    {   
        OPENCVLIB_API void OPENCVLIB_CALL licenseplate(const char * & __result,const void * s);

        OPENCVLIB_API void OPENCVLIB_CALL edge_detect(const char * & __result,const void * s,long long threshold);
        
        OPENCVLIB_API void OPENCVLIB_CALL gaussblur(const char * & __result,const void * s,long long scale);

        OPENCVLIB_API void OPENCVLIB_CALL grayscale(const char * & __result,const void * s);

        OPENCVLIB_API void OPENCVLIB_CALL resize(const char * & __result,const void * s,double fx, double fy);

        OPENCVLIB_API void OPENCVLIB_CALL rotate_img(const char * & __result,const void * s,double angle);
 
        OPENCVLIB_API void OPENCVLIB_CALL threshold_img(const char * & __result,const void * s,double threshval,double maxval,long long type);

        OPENCVLIB_API void OPENCVLIB_CALL translate_img(const char * & __result,const void * s,double x, double y);
        static void CannyThreshold(int, void*);
        void displayResults (cv::Mat src, cv::Mat dest);
    }
}

#endif

