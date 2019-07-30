/*Base CPP File*/
#include "opencvlib.hpp"

//==============================================================================
// Service Library Code
//==============================================================================

namespace OPENCVLib
{
    using namespace cv;
    using namespace std;

    /*License Plate Detection function */
    OPENCVLIB_API void OPENCVLIB_CALL licenseplate(size32_t & __lenResult,char *  & __result, size32_t lenData,const void * data)
    {
        bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();
        
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);
       
        cv::Mat imgOriginalScene;   
        std::string result;
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        
        /* Training KNN Modules */
        if (!blnKNNTrainingSuccessful){
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }
        
        /*Reading the image from the supplied path */    
        imgOriginalScene = cv::imdecode(Mat(img_data), -1);  
        if (imgOriginalScene.empty()) 
        {                                
            __lenResult=4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);                                                                                       
        }

        /*Detecting Plates */
        std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          

        /*Detecting Chars in Plates */
        vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               

        /*Checking for possible plates */        
        if (vectorOfPossiblePlates.empty()) 
        {                                                  
            __lenResult=4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }
        else 
        {                                                                                                                                                  
            std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);
            PossiblePlate licPlate = vectorOfPossiblePlates.front();
            /*Setting the result based on licPlate output */
            if (licPlate.strChars.length() == 0) 
            {                                                          
                __lenResult=4;
                __result = reinterpret_cast<char*>(rtlMalloc(4));
                memcpy(__result,fail,4);                                                                       
            }
            else {
                result=licPlate.strChars.c_str();
                __lenResult=licPlate.strChars.length();
                __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
                char *c = new char[__lenResult];
                std::copy(result.begin(), result.end(), c);
                memcpy(__result,c,__lenResult);  
            }       
        }
    }
    /* ============================================================================================================================ */
    
    /*Function to perform gaussian blur of an image */
    OPENCVLIB_API void OPENCVLIB_CALL gaussblur(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data, long long scale)
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);
        
        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        cv::Mat image = cv::imdecode(Mat(img_data), -1); 
        if (image.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }

        /*Performing Blur */
        Mat image_blurred_with_nxn_kernel;
        GaussianBlur(image, image_blurred_with_nxn_kernel, Size(scale, scale), 0);

        /*Storing Result*/
        imencode(".jpg",image_blurred_with_nxn_kernel,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(image, image_blurred_with_nxn_kernel);       
    }

    /*Function to perform greyscale of an image */
    OPENCVLIB_API void OPENCVLIB_CALL grayscale(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data)
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        cv::Mat image = cv::imdecode(Mat(img_data), -1); 
        if (image.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }

        /*convert RGB image to gray*/
        Mat gray;
        cvtColor(image, gray, CV_BGR2GRAY);

        /*Storing result*/
        imencode(".jpg",gray,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(image, gray);
    }

    /*Function to perform resizing of an image */
    OPENCVLIB_API void OPENCVLIB_CALL resize(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data,double fx, double fy)
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        cv::Mat image = cv::imdecode(Mat(img_data), -1); 
        if (image.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }

        /*Performing resize*/
        Mat out;
        resize(image, out, Size(),fx,fy);

        /*Storing result*/
        imencode(".jpg",out,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(image, out);
    }

    /*Function to perform rotation of an image */
    OPENCVLIB_API void OPENCVLIB_CALL rotate_img(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data,double angle)
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        cv::Mat src = cv::imdecode(Mat(img_data), -1); 
        if (src.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }
        
        /* Get rotation matrix */
        cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);

        /*Obtain Bounding Ractangle*/
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();

        /*Adjust transformation matrix */
        rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
        rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

        /*Perform Rotation */
        cv::Mat dst;
        cv::warpAffine(src, dst, rot, bbox.size());

        /*Storing result*/
        imencode(".jpg",dst,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(src, dst);
    } 

    /*Function to thresholding of an image */
    OPENCVLIB_API void OPENCVLIB_CALL threshold_img(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data,double threshval, double maxval=255,long long type=0 )
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        cv::Mat image = cv::imdecode(Mat(img_data), -1); 
        if (image.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }

        /*Performing thresholding */
        Mat res; 
        threshold(image,res, threshval,maxval,type);

        /*Storing result*/
        imencode(".jpg",res,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(image, res);
    }

    /*Function to perform translation of an image */
    OPENCVLIB_API void OPENCVLIB_CALL translate_img(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data,double x,double y)
    {
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        /*Loading source */
        Mat src, warp_dst;
        Mat warp_mat = Mat::eye(2, 3, CV_64F);
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */    
        src = cv::imdecode(Mat(img_data), -1); 
        if (src.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }
        
        /* Setting transform matrix */
        warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
        warp_mat.at<double>(0,2) = x; 
        warp_mat.at<double>(1,2) = y;

        /*Applying Affine Transform */
        warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

        /*Storing result*/
        imencode(".jpg",warp_dst,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(src, warp_dst);
    }

    /* ============================================================================================================================ */

    /*Edge Detection function */
    OPENCVLIB_API long long OPENCVLIB_CALL edge_detect(size32_t & __lenResult, void * & __result,size32_t lenData,const void * data, long long threshold)
    {  
        char fail[4];
        memset(fail,0,sizeof(fail));
        memcpy(fail,"Fail",4);

        Mat src, src_gray;
        Mat dst, detected_edges;
        const int setThresold = threshold > 100 ? 100 : threshold;
        const int ratio = 3;
        const int kernel_size = 3;

        /*Loading source */
        std::vector<char> img_data((char *) data, (char *)data + lenData);
        std::vector<uchar> buf;
        
        /*Decoding image */
        src = cv::imdecode(Mat(img_data), -1); 
        if (src.empty())
        {
            __lenResult = 4;
            __result = reinterpret_cast<char*>(rtlMalloc(4));
            memcpy(__result,fail,4);
        }

        /*Creating edge detected output*/
        dst.create( src.size(), src.type() );
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        blur( src_gray, detected_edges, Size(3,3) );
        Canny( detected_edges, detected_edges, threshold, threshold*ratio, kernel_size );
        dst = Scalar::all(0);
        src.copyTo( dst, detected_edges);

        /*Storing Result*/
        imencode(".jpg",dst,buf);
        __lenResult = buf.size();
        __result = reinterpret_cast<char*>(rtlMalloc(__lenResult));
        char* c = new char[__lenResult +1];
        std::copy(buf.begin(), buf.end(), c);
        memcpy(__result,c,__lenResult); 

        /*Display Section (comment before running on cluster)*/
        // displayResults(src, dst);       
    }

    /* ============================================================================================================================ */
    /*Helper functions*/

    void displayResults(cv::Mat input, cv::Mat output)
    {
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", input );
        namedWindow( "Display output window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display output window", output );
        waitKey(0);
    }
} 

