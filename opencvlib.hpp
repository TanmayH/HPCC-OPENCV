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
        OPENCVLIB_API void OPENCVLIB_CALL licenseplate(const char *  & __result,const char * path);

        OPENCVLIB_API long long OPENCVLIB_CALL edge_detect(const char * path);
        
        OPENCVLIB_API long long OPENCVLIB_CALL gaussblur(const char * path,const char * dest,long long scale);

        OPENCVLIB_API long long OPENCVLIB_CALL grayscale(const char * path, const char * dest);

        OPENCVLIB_API long long OPENCVLIB_CALL resize(const char * path,const char * dest,double fx, double fy);

        OPENCVLIB_API long long OPENCVLIB_CALL rotate_img(const char * path,const char * dest,double angle);
 
        OPENCVLIB_API long long OPENCVLIB_CALL threshold_img(const char * path,const char * dest,double threshval,double maxval,long long type);

        OPENCVLIB_API long long OPENCVLIB_CALL translate_img(const char * path,const char * dest,double x, double y);
        
        void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
        void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
        static void CannyThreshold(int, void*);
    }
}

#endif

