#include "DetectChars.h"

cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

bool loadKNNDataAndTrainKNN(void) 
{  
    cv::Mat matClassificationInts;             
    cv::FileStorage fsClassifications("./xml/classifications.xml", cv::FileStorage::READ);        
    if (fsClassifications.isOpened() == false) {                                                      
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        
        return(false);                                                                                  
    }

    fsClassifications["classifications"] >> matClassificationInts;          
    fsClassifications.release();                                           
    cv::Mat matTrainingImagesAsFlattenedFloats;       
    cv::FileStorage fsTrainingImages("./xml/images.xml", cv::FileStorage::READ);              
    if (fsTrainingImages.isOpened() == false) 
    {                                                 
        std::cout << "error, unable to open training images file, exiting program\n\n";       
        return(false); 
    }                                                                         
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;          
    fsTrainingImages.release();                                               
    kNearest->setDefaultK(1);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    return true;
}

std::vector<PossiblePlate> detectCharsInPlates(std::vector<PossiblePlate> &vectorOfPossiblePlates) 
{
    int intPlateCounter = 0;				
    cv::Mat imgContours;
    std::vector<std::vector<cv::Point> > contours;
    cv::RNG rng;

    if (vectorOfPossiblePlates.empty()) 
    {               
        return(vectorOfPossiblePlates);                 
    }

    for (auto &possiblePlate : vectorOfPossiblePlates) 
    {            

        preprocess(possiblePlate.imgPlate, possiblePlate.imgGrayscale, possiblePlate.imgThresh); 

        #ifdef SHOW_STEPS
                cv::imshow("5a", possiblePlate.imgPlate);
                cv::imshow("5b", possiblePlate.imgGrayscale);
                cv::imshow("5c", possiblePlate.imgThresh);
        #endif	

        
        cv::resize(possiblePlate.imgThresh, possiblePlate.imgThresh, cv::Size(), 1.6, 1.6);
        cv::threshold(possiblePlate.imgThresh, possiblePlate.imgThresh, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

        #ifdef SHOW_STEPS
                cv::imshow("5d", possiblePlate.imgThresh);
        #endif	
        
        std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh);

        #ifdef SHOW_STEPS
                imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);
                contours.clear();

                for (auto &possibleChar : vectorOfPossibleCharsInPlate) 
                {
                    contours.push_back(possibleChar.contour);
                }

                cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

                cv::imshow("6", imgContours);
        #endif	
                std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

        #ifdef SHOW_STEPS
                imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

                contours.clear();

                for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) 
                {
                    int intRandomBlue = rng.uniform(0, 256);
                    int intRandomGreen = rng.uniform(0, 256);
                    int intRandomRed = rng.uniform(0, 256);

                    for (auto &matchingChar : vectorOfMatchingChars) 
                    {
                        contours.push_back(matchingChar.contour);
                    }
                    cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
                }
                cv::imshow("7", imgContours);
        #endif	

        if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) 
        {                
            #ifdef SHOW_STEPS
                        std::cout << "chars found in plate number " << intPlateCounter << " = (none), click on any image and press a key to continue . . ." << std::endl;
                        intPlateCounter++;
                        cv::destroyWindow("8");
                        cv::destroyWindow("9");
                        cv::destroyWindow("10");
                        cv::waitKey(0);
            #endif	
            possiblePlate.strChars = "";            
            continue;                              
        }

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) 
        {                                         
            std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);     
            vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                   
        }

        #ifdef SHOW_STEPS
                imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

                for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) 
                {
                    int intRandomBlue = rng.uniform(0, 256);
                    int intRandomGreen = rng.uniform(0, 256);
                    int intRandomRed = rng.uniform(0, 256);

                    contours.clear();

                    for (auto &matchingChar : vectorOfMatchingChars) 
                    {
                        contours.push_back(matchingChar.contour);
                    }
                    cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
                }
                cv::imshow("8", imgContours);
        #endif	

        
        unsigned int intLenOfLongestVectorOfChars = 0;
        unsigned int intIndexOfLongestVectorOfChars = 0;
    
        for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) 
        {
            if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) 
            {
                intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                intIndexOfLongestVectorOfChars = i;
            }
        }
    
        std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

        #ifdef SHOW_STEPS
                imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

                contours.clear();

                for (auto &matchingChar : longestVectorOfMatchingCharsInPlate) 
                {
                    contours.push_back(matchingChar.contour);
                }
                cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

                cv::imshow("9", imgContours);
        #endif	

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestVectorOfMatchingCharsInPlate);

        #ifdef SHOW_STEPS
                std::cout << "chars found in plate number " << intPlateCounter << " = " << possiblePlate.strChars << ", click on any image and press a key to continue . . ." << std::endl;
                intPlateCounter++;
                cv::waitKey(0);
        #endif	

    }   

    #ifdef SHOW_STEPS
        std::cout << std::endl << "char detection complete, click on any image and press a key to continue . . ." << std::endl;
        cv::waitKey(0);
    #endif

    return(vectorOfPossiblePlates);
}

std::vector<PossibleChar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh) 
{
    std::vector<PossibleChar> vectorOfPossibleChars;                           
    cv::Mat imgThreshCopy;
    std::vector<std::vector<cv::Point> > contours;
    imgThreshCopy = imgThresh.clone();				
    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        
    for (auto &contour : contours) 
    {                           
        PossibleChar possibleChar(contour);
        if (checkIfPossibleChar(possibleChar)) 
        {               
            vectorOfPossibleChars.push_back(possibleChar);      
        }
    }
    return(vectorOfPossibleChars);
}

bool checkIfPossibleChar(PossibleChar &possibleChar) 
{

    if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
        possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
        MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) 
    {
        return(true);
    }
    else 
    {
        return(false);
    }
}

std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar> &vectorOfPossibleChars) 
{

    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;            
    for (auto &possibleChar : vectorOfPossibleChars) 
    {                 
                                                                    
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);         

                                                            
        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) 
        {
            continue;                     
        }
        
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);           
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;

        for (auto &possChar : vectorOfPossibleChars) 
        {
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end())
            {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
    
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

    
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);	

        for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) 
        {      
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               
        }

        break;		
    }

    return(vectorOfVectorsOfMatchingChars);
}


std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars) 
{
    std::vector<PossibleChar> vectorOfMatchingChars;                
    for (auto &possibleMatchingChar : vectorOfChars) 
    {             

                                                                    
        if (possibleMatchingChar == possibleChar) 
        {
            
            continue;          
        }
    
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&
            dblChangeInArea < MAX_CHANGE_IN_AREA &&
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) 
        {
            vectorOfMatchingChars.push_back(possibleMatchingChar);      
        }   
    }

    return(vectorOfMatchingChars);          
}


double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) 
{
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);
    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}


double angleBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) 
{
    double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
    double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);
    double dblAngleInRad = atan(dblOpp / dblAdj);
    double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);
    return(dblAngleInDeg);
}


std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar> &vectorOfMatchingChars) 
{
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);
    
    for (auto &currentChar : vectorOfMatchingChars) 
    {
        for (auto &otherChar : vectorOfMatchingChars) 
        {
            if (currentChar != otherChar) 
            {                        
                                                                
                if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) 
                {
                
                    if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) 
                    {
                    
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                    
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) 
                        {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       
                        }
                    }
                    else 
                    {        
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
                    
                        if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) 
                        {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         
                        }
                    }
                }
            }
        }
    }

    return(vectorOfMatchingCharsWithInnerCharRemoved);
}


std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars) 
{
    std::string strChars;               
    cv::Mat imgThreshColor;
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);
    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);      
    
    for (auto &currentChar : vectorOfMatchingChars) 
    {          
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       
        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);               
        cv::Mat imgROI = imgROItoBeCloned.clone();     
        cv::Mat imgROIResized;
    
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         
        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);      
        cv::Mat matCurrentChar(0, 0, CV_32F);                  
        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     
        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);      
        strChars = strChars + char(int(fltCurrentChar));        
    }

    #ifdef SHOW_STEPS
        cv::imshow("10", imgThreshColor);
    #endif	

    return(strChars);               
}