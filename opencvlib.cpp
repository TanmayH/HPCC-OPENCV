/*Base CPP File*/
#include "opencvlib.hpp"

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

    /*License Plate Detection function */
    OPENCVLIB_API void OPENCVLIB_CALL licenseplate(size32_t & __lenResult,char *  & __result,const char * path)
    {
        bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();
        cv::Mat imgOriginalScene;   
        std::string fail = "Fail";
        std::string result;

        /* Training KNN Modules */
        if (!blnKNNTrainingSuccessful){
            std::cout << "Error: KNN traning was not successful";
        }   

        /*Reading the image from the supplied path */    
        imgOriginalScene = cv::imread(path);   

        if (imgOriginalScene.empty()) 
        {                             
            std::cout << "error: image not read from file\n\n";     
            __lenResult=4;
            __result=const_cast<char*>(fail.c_str());                                                                                       
        }

        /*Detecting Plates */
        std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          

        /*Detecting Chars in Plates */
        vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               

        /*Original Image*/
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

            /*Displaying plates as overlay*/
            cv::imshow("imgPlate", licPlate.imgPlate);            
            cv::imshow("imgThresh", licPlate.imgThresh);

            if (licPlate.strChars.length() == 0) 
            {                                                     
                std::cout << std::endl << "no characters were detected" << std::endl << std::endl;      
                __lenResult=4;
                __result=const_cast<char*>(fail.c_str());                                                                        
            }

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate);

            /*Setting Result*/
            __lenResult=licPlate.strChars.length();
            result=licPlate.strChars.c_str();

            char *c = new char[__lenResult + 1];
            std::copy(result.begin(), result.end(), c);
            c[__lenResult] = '\0';
            __result=c;
                
            std::cout << std::endl << "Press any key to close window.." << std::endl;

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);            

            cv::imshow("car", imgOriginalScene);                      

            cv::waitKey(0);        
        }

    }

    /*Function to draw Rectangular Bounds around Plates */
    void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) 
    {
        cv::Point2f p2fRectPoints[4];

        licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);           

        for (int i = 0; i < 4; i++) 
        {                                       
            cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
        }
    }

    /*Function to write License Plate Chars on Image*/ 
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

    /* ============================================================================================================================ */
    
    /*Edge Detection function */
    OPENCVLIB_API long long OPENCVLIB_CALL edge_detect(const char * path)
    {   
        /*Load the image */
        src = imread( path, IMREAD_COLOR );
        if( src.empty() )
        {
            std::cout << "Could not open or find the image!\n" << std::endl;
            return 0;
        }

        dst.create( src.size(), src.type() );
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        namedWindow( window_name, WINDOW_AUTOSIZE );

        /*Setting the Threshold */
        createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
        CannyThreshold(0, 0);
        std::cout << std::endl << "Press any key to close window.." << std::endl;

        waitKey(0);
        return 1;
    }

    /*Threshold Setter */
    static void CannyThreshold(int, void*)
    {
        blur( src_gray, detected_edges, Size(3,3) );
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        dst = Scalar::all(0);
        src.copyTo( dst, detected_edges);
        imshow( window_name, dst );
    }

    /* ============================================================================================================================ */
    
    /*Function to perform gaussian blur of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL gaussblur(const char * path,const char * dest, long long scale)
    {
        /*Loading source */
        Mat image = imread(path);
        if (image.empty())
        {
            cout << "Could not open or find the image" << endl;
            return -1;
        }
        
        /*Performing Blur */
        Mat image_blurred_with_nxn_kernel;
        GaussianBlur(image, image_blurred_with_nxn_kernel, Size(scale, scale), 0);

        
        String window_name = "Image";
        String window_name_blurred_with_nxn_kernel = "Image after Gaussian Blur" ;

        /*Displaying Original and blurred image*/
        namedWindow(window_name);
        namedWindow(window_name_blurred_with_nxn_kernel);

        
        imshow(window_name, image);
        imshow(window_name_blurred_with_nxn_kernel, image_blurred_with_nxn_kernel);

        /*Writing result to destination */
        imwrite(dest,image_blurred_with_nxn_kernel);

        std::cout << std::endl << "Press any key to close window.." << std::endl;

        waitKey(0); 

        destroyAllWindows(); 

        return 0;
    }

    /*Function to perform greyscale of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL grayscale(const char * path, const char * dest)
    {
        /*Loading source */
        Mat image = imread(path);
        if(! image.data )                             
        {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
        }
        Mat gray;
    
        /*convert RGB image to gray*/
        cvtColor(image, gray, CV_BGR2GRAY);

        /*Display Original and greyscaled results*/
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", image );                 
    
        namedWindow( "Result window", CV_WINDOW_AUTOSIZE );   
        imshow( "Result window", gray );

        /*Writing result to destination */
        imwrite(dest,gray);

        std::cout << std::endl << "Press any key to close window.." << std::endl;
    
        waitKey(0);

        return 0;

    }

    /*Function to perform resizing of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL resize(const char * path, const char * dest,double fx, double fy)
    {
        /*Loading source */
        Mat image = imread(path);
        if(! image.data )                             
        {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
        }
        Mat out;

        /*Performing resize*/
        resize(image, out, Size(),fx,fy);
        
        /*Displaying Results*/
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", out );  

        /*Writing result to destination */
        imwrite(dest,out);

        std::cout << std::endl << "Press any key to close window.." << std::endl;               
    
        waitKey(0);

        return 0;
    }

    /*Function to perform rotation of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL rotate_img(const char * path,const char * dest,double angle)
    {
        cv::Mat src = cv::imread(path, CV_LOAD_IMAGE_UNCHANGED);
        
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

        /*Display Results */
        namedWindow( "Result window", CV_WINDOW_AUTOSIZE );   
        imshow( "Result window", dst );

        /*Writing Results to destination*/
        cv::imwrite(dest, dst);

        std::cout << std::endl << "Press any key to close window.." << std::endl;             
 
        waitKey(0);

        return 0;

    } 

    /*Function to thresholding of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL threshold_img(const char * path, const char * dest,double threshval, double maxval=255,long long type=0 )
    {
        /*Loading source */
        Mat image = imread(path,0);
        if(! image.data )                             
        {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
        }
        Mat res; 

        /*Performing thresholding */
        threshold(image,res, threshval,maxval,type);
 
        /*Displaying Results */
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
        imshow( "Display window", res );  

        /*Writing result to destination */
        imwrite(dest,res);

        std::cout << std::endl << "Press any key to close window.." << std::endl;               
 
        waitKey(0);
        return 0;
    }

    /*Function to perform translation of an image */
    OPENCVLIB_API long long OPENCVLIB_CALL translate_img(const char * path,const char * dest, double x,double y)
    {
        /*Loading source */
        Mat src, warp_dst;
        Mat warp_mat = Mat::eye(2, 3, CV_64F);

        src = imread( path, 1 );
        if(! src.data )                             
        {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
        }
        
        /* Setting transform matrix */
        warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
        warp_mat.at<double>(0,2) = x; 
        warp_mat.at<double>(1,2) = y;

        /*Applying Affine Transform */
        warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

        /*Displaying Results */
        namedWindow( "Source Window", CV_WINDOW_AUTOSIZE );
        imshow( "Source Window", src );

        namedWindow( "Warp Window", CV_WINDOW_AUTOSIZE );
        imshow( "Warp Window", warp_dst );

        /*Writing result to destination */
        imwrite(dest,warp_dst);

        std::cout << std::endl << "Press any key to close window.." << std::endl;
        waitKey(0);

        return 0;

    }

    /* ============================================================================================================================ */

} 

