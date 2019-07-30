#include "DetectPlates.h"

std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene) 
{
    std::vector<PossiblePlate> vectorOfPossiblePlates;			
    cv::Mat imgGrayscaleScene;
    cv::Mat imgThreshScene;
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    cv::RNG rng;

    cv::destroyAllWindows();

    #ifdef SHOW_STEPS
        cv::imshow("0", imgOriginalScene);
    #endif	

    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene); 

    #ifdef SHOW_STEPS
        cv::imshow("1a", imgGrayscaleScene);
        cv::imshow("1b", imgThreshScene);
    #endif	


    std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);

    #ifdef SHOW_STEPS
        std::cout << "step 2 - vectorOfPossibleCharsInScene.Count = " << vectorOfPossibleCharsInScene.size() << std::endl;        
        imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);
        std::vector<std::vector<cv::Point> > contours;

        for (auto &possibleChar : vectorOfPossibleCharsInScene) 
        {
            contours.push_back(possibleChar.contour);
        }
        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);
        cv::imshow("2b", imgContours);
    #endif	

    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

    #ifdef SHOW_STEPS
        std::cout << "step 3 - vectorOfVectorsOfMatchingCharsInScene.size() = " << vectorOfVectorsOfMatchingCharsInScene.size() << std::endl;        

        imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) 
        {
            int intRandomBlue = rng.uniform(0, 256);
            int intRandomGreen = rng.uniform(0, 256);
            int intRandomRed = rng.uniform(0, 256);

            std::vector<std::vector<cv::Point> > contours;

            for (auto &matchingChar : vectorOfMatchingChars) 
            {
                contours.push_back(matchingChar.contour);
            }
            cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
        }
        cv::imshow("3", imgContours);
    #endif	

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) 
    {                    
        PossiblePlate possiblePlate = extractPlate(imgOriginalScene, vectorOfMatchingChars);        
        if (possiblePlate.imgPlate.empty() == false) 
        {                                              
            vectorOfPossiblePlates.push_back(possiblePlate);                                      
        }
    }  

    #ifdef SHOW_STEPS
        std::cout << std::endl;
        cv::imshow("4a", imgContours);

        for (unsigned int i = 0; i < vectorOfPossiblePlates.size(); i++) 
        {
            cv::Point2f p2fRectPoints[4];

            vectorOfPossiblePlates[i].rrLocationOfPlateInScene.points(p2fRectPoints);

            for (int j = 0; j < 4; j++) 
            {
                cv::line(imgContours, p2fRectPoints[j], p2fRectPoints[(j + 1) % 4], SCALAR_RED, 2);
            }
            cv::imshow("4a", imgContours);

            std::cout << "possible plate " << i << ", click on any image and press a key to continue . . ." << std::endl;

            cv::imshow("4b", vectorOfPossiblePlates[i].imgPlate);
            cv::waitKey(0);
        }
        std::cout << std::endl << "plate detection complete, click on any image and press a key to begin char recognition . . ." << std::endl << std::endl;
        cv::waitKey(0);
    #endif	

    return vectorOfPossiblePlates;
}


std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat &imgThresh) 
{
    std::vector<PossibleChar> vectorOfPossibleChars;            
    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);
    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);       
    for (unsigned int i = 0; i < contours.size(); i++) 
    {               
        #ifdef SHOW_STEPS
                cv::drawContours(imgContours, contours, i, SCALAR_WHITE);
        #endif	
        PossibleChar possibleChar(contours[i]);

        if (checkIfPossibleChar(possibleChar)) 
        {                
            intCountOfPossibleChars++;                          
            vectorOfPossibleChars.push_back(possibleChar);      
        }
    }

    #ifdef SHOW_STEPS
        std::cout << std::endl << "contours.size() = " << contours.size() << std::endl;                         
        std::cout << "step 2 - intCountOfValidPossibleChars = " << intCountOfPossibleChars << std::endl;        
        cv::imshow("2a", imgContours);
    #endif	

    return(vectorOfPossibleChars);
}


PossiblePlate extractPlate(cv::Mat &imgOriginal, std::vector<PossibleChar> &vectorOfMatchingChars) 
{
    PossiblePlate possiblePlate;            
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY);

    
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto &matchingChar : vectorOfMatchingChars) 
    {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);


    double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);

    
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());           

                                                                                            
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);

    possiblePlate.imgPlate = imgCropped;            
    return(possiblePlate);
}