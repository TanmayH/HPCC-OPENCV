
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string> // for string class 
// #include "Main.h"
#include "opencvlib.hpp"

/*Try to bring out Mat data type,vector of possible plates, imshow*/


//==============================================================================
// Service Library Code
//==============================================================================

namespace OPENCVLib
{
    using namespace cv;
    using namespace std;

    Mat src, src_gray;
    Mat dst, detected_edges;
    int lowThreshold = 0;
    const int max_lowThreshold = 100;
    const int ratio = 3;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";
    // using namespace cv::xfeatures2d;
    OPENCVLIB_API bool OPENCVLIB_CALL loadKNNDataAndTrainKNN2()
    {
        bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();
        return blnKNNTrainingSuccessful;
    }

    OPENCVLIB_API void OPENCVLIB_CALL licenseplate(size32_t & __lenResult,char *  & __result,const char * path)
    {
        
        cv::Mat imgOriginalScene;   
        
        std::string fail = "Fail";
        std::string result;       
        imgOriginalScene = cv::imread(path);   

        if (imgOriginalScene.empty()) 
        {                             
            std::cout << "error: image not read from file\n\n";     
            __lenResult=4;
            __result=const_cast<char*>(fail.c_str());                                                                                       
        }

        std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          

        vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               

        cv::imshow("car", imgOriginalScene);          

        if (vectorOfPossiblePlates.empty()) 
        {                                              
            std::cout << std::endl << "no license plates were detected" << std::endl;     
            __lenResult=4;
            __result=const_cast<char*>(fail.c_str());
        }
        else 
        {                                                                           
                                                                                        
            std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);
        
            PossiblePlate licPlate = vectorOfPossiblePlates.front();

            cv::imshow("imgPlate", licPlate.imgPlate);            
            cv::imshow("imgThresh", licPlate.imgThresh);

            if (licPlate.strChars.length() == 0) 
            {                                                     
                std::cout << std::endl << "no characters were detected" << std::endl << std::endl;      
                __lenResult=4;
                __result=const_cast<char*>(fail.c_str());                                                                        
            }

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate);

            __lenResult=licPlate.strChars.length();
            result=licPlate.strChars.c_str();

            char *c = new char[__lenResult + 1];
            std::copy(result.begin(), result.end(), c);
            c[__lenResult] = '\0';
            __result=c;
            

            std::cout << std::endl << "license plate read from image = " << licPlate.strChars.c_str() << std::endl;     
            std::cout << std::endl << "-----------------------------------------" << std::endl;

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);            

            cv::imshow("car", imgOriginalScene);                      

            cv::imwrite("processed_car_image.png", imgOriginalScene);

            cv::waitKey(0);        
        }

    }
    
    
    // OPENCVLIB_API void OPENCVLIB_CALL feature_match(size32_t & __lenResult,char *  & __result,const char * path_src,const char * path_dest)
    // {
    //     std::string fail = "Fail";
    //     std::string result;  
        
  
    //     Mat img_1 = imread( path_src, IMREAD_GRAYSCALE );
    //     Mat img_2 = imread( path_dest, IMREAD_GRAYSCALE );
    //     if( !img_1.data || !img_2.data )
    //     { 
    //         std::cout<< " --(!) Error reading images " << std::endl; 
    //         __lenResult=4;
    //         __result=const_cast<char*>(fail.c_str()); 
    //     }

    //     //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    //     int minHessian = 400;
    //     Ptr<SURF> detector = SURF::create();
    //     detector->setHessianThreshold(minHessian);
    //     std::vector<KeyPoint> keypoints_1, keypoints_2;
    //     Mat descriptors_1, descriptors_2;
    //     detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
    //     detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );
        
    //     //-- Step 2: Matching descriptor vectors using FLANN matcher
    //     FlannBasedMatcher matcher;
    //     std::vector< DMatch > matches;
    //     matcher.match( descriptors_1, descriptors_2, matches );
    //     double max_dist = 0; double min_dist = 100;
        
    //     //-- Quick calculation of max and min distances between keypoints
    //     for( int i = 0; i < descriptors_1.rows; i++ )
    //     { double dist = matches[i].distance;
    //         if( dist < min_dist ) min_dist = dist;
    //         if( dist > max_dist ) max_dist = dist;
    //     }
    //     printf("-- Max dist : %f \n", max_dist );
    //     printf("-- Min dist : %f \n", min_dist );
        
    //     //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //     //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //     //-- small)
    //     //-- PS.- radiusMatch can also be used here.
    //     std::vector< DMatch > good_matches;
    //     for( int i = 0; i < descriptors_1.rows; i++ )
    //     { 
    //         if( matches[i].distance <= max(2*min_dist, 0.02) )
    //         { 
    //             good_matches.push_back( matches[i]); 
    //         }
    //     }
        
    //     //-- Draw only "good" matches
    //     Mat img_matches;
    //     drawMatches( img_1, keypoints_1, img_2, keypoints_2,
    //                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
    //                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
    //     //-- Show detected matches
    //     imshow( "Good Matches", img_matches );
        
    //     for( int i = 0; i < (int)good_matches.size(); i++ )
    //     { 
    //         printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); 
    //     }
    //     waitKey(0);

    // }

    OPENCVLIB_API long long OPENCVLIB_CALL edge_detect(const char * path)
    {   
        src = imread( path, IMREAD_COLOR ); // Load an image
        if( src.empty() )
        {
            std::cout << "Could not open or find the image!\n" << std::endl;
            return 0;
        }
        dst.create( src.size(), src.type() );
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        namedWindow( window_name, WINDOW_AUTOSIZE );
        createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
        CannyThreshold(0, 0);
        waitKey(0);
        return 1;
    }

    static void CannyThreshold(int, void*)
    {
        blur( src_gray, detected_edges, Size(3,3) );
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        dst = Scalar::all(0);
        src.copyTo( dst, detected_edges);
        imshow( window_name, dst );
    }



    void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) 
    {
        cv::Point2f p2fRectPoints[4];

        licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);           

        for (int i = 0; i < 4; i++) 
        {                                       
            cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
        }
    }


    void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) 
    {
        cv::Point ptCenterOfTextArea;                   
        cv::Point ptLowerLeftTextOrigin;                

        int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                              
        double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            
        int intFontThickness = (int)std::round(dblFontScale * 1.5);             
        int intBaseline = 0;

        cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);      

        ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         

        if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      
                                                                                                
            ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
        }
        else {                                                                                
            ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
        }

        ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           
        ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          
        cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
    }


    
    
    OPENCVLIB_API long long OPENCVLIB_CALL gaussblur(const char * path)
    {
        Mat image = imread(path);

    
        if (image.empty())
        {
            cout << "Could not open or find the image" << endl;
            cin.get(); //wait for any key press
            return -1;
        }


        Mat image_blurred_with_3x3_kernel;
        GaussianBlur(image, image_blurred_with_3x3_kernel, Size(9, 9), 0);

        
        Mat image_blurred_with_5x5_kernel;
        GaussianBlur(image, image_blurred_with_5x5_kernel, Size(17, 17), 0);

        
        String window_name = "Car";
        String window_name_blurred_with_3x3_kernel = "Car Blurred with 9 X 9 Gaussian Kernel";
        String window_name_blurred_with_5x5_kernel = "Car Blurred with 17 X 17 Gaussian Kernel";

        
        namedWindow(window_name);
        namedWindow(window_name_blurred_with_3x3_kernel);
        namedWindow(window_name_blurred_with_5x5_kernel);

        
        imshow(window_name, image);
        imshow(window_name_blurred_with_3x3_kernel, image_blurred_with_3x3_kernel);
        imshow(window_name_blurred_with_5x5_kernel, image_blurred_with_5x5_kernel);

        waitKey(0); 

        destroyAllWindows(); 

        return 0;
    }


    OPENCVLIB_API long long OPENCVLIB_CALL grayscale(const char * path)
    {
        //TODO: pass other parameters for hsv,grayscale and all
            Mat image = imread(path);
        if(! image.data )                             
        {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
        }
        Mat gray;
    
        // convert RGB image to gray
        cvtColor(image, gray, CV_BGR2GRAY);
    
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", image );                 
    
        namedWindow( "Result window", CV_WINDOW_AUTOSIZE );   
        imshow( "Result window", gray );
    
        waitKey(0);
    

        return 0;

    }


    OPENCVLIB_API long long OPENCVLIB_CALL resize(const char * path, double fx, double fy)
    {
            //TODO: pass other parameters for hsv,grayscale and all
            Mat image = imread(path);
        if(! image.data )                             
        {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
        }
        Mat out;
    
        // convert RGB image to gray
        resize(image, out, Size(),fx,fy);
    
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", out );                 
    

    
        waitKey(0);
    

        return 0;




    }

    OPENCVLIB_API long long OPENCVLIB_CALL rotate_img(const char * path,double angle,double scale, double x=-1.0, double y=-1.0)
    {

          //TODO: pass other parameters for hsv,grayscale and all
        Mat image = imread(path);
        if(! image.data )                             
            {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
        }
        Mat gray;
    
        if(x==-1.0){
            x=image.cols/2;
        }
        if(y==-1.0){
            y=image.rows/2;
        }

       // convert RGB image to gray
        Mat dst1;
    
        Mat res = getRotationMatrix2D(Point(x,y), angle, scale);
        warpAffine(image,dst1,res,image.size()); 
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", dst1);                 
    
   
 
        waitKey(0);
 

        return 0;

    } 

    OPENCVLIB_API long long OPENCVLIB_CALL threshold_img(const char * path, double threshval, double maxval=255,long long type=0 )
    {
          //TODO: pass other parameters for various 5 types of thresholding

        // src_gray: Our input image
        // dst: Destination (output) image
        // threshold_value: The thresh value with respect to which the thresholding operation is made
        // max_BINARY_value: The value used with the Binary thresholding operations (to set the chosen pixels)
        // threshold_type: One of the 5 thresholding operations. 
        Mat image = imread(path,0);
        if(! image.data )                             
       {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
       }
        Mat res;
        /* 0: Binary
        1: Binary Inverted
        2: Threshold Truncated
        3: Threshold to Zero
        4: Threshold to Zero Inverted
        */

        threshold(image,res, threshval,maxval,type);
 
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", res );                 
 

 
        waitKey(0);
 

        return 0;


    }

    OPENCVLIB_API long long OPENCVLIB_CALL translate_img(const char * path, double rs,double cs)
    {
          //TODO: pass other parameters for hsv,grayscale and all
          //https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
        Mat image = imread(path);
        if(! image.data )                             
       {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
       }
 
       // convert RGB image to gray
        Mat out = Mat::zeros(image.size(), image.type());

        Mat par = Mat(2, 3, CV_64FC1); // Allocate memory
        par.at<double>(0,0)=  1;  //p1
        par.at<double>(1,0)=  0;  //p2;
        par.at<double>(0,1)= 0; //p3;
        par.at<double>(1,1)= 1;  //p4;
        par.at<double>(0,2)= cs;   //p5;
        par.at<double>(1,2)= rs;//p6;
        warpAffine(image,par,out,image.size());

        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", out );                 
 

 
         waitKey(0);
 

        return 0;

    }

} // namespace

