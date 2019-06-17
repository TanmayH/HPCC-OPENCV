IMPORT opencv.opencv;

/*Test case 1 for license plate detection*/
res := opencv.GETLICENSEPLATE('./test_images/test1.jpg');


/*Test Case 2 for individual functions*/
// res:=opencv.GAUSSBLUR('./test_images/test2.jpg','./Modified_Images/BlurImage.jpg',21);
// res:=opencv.GRAYSCALE('./test_images/test2.jpg','./Modified_Images/GrayImage.jpg');
// res:=opencv.THRESH_HOLD('./test_images/test2.jpg','./Modified_Images/ThresholdImage.jpg',10);
// res:=opencv.RESIZE('./test_images/test2.jpg','./Modified_Images/ResizedImage.jpg',0.15,0.95);
// res:=opencv.TRANSLATE('./test_images/test2.jpg','./Modified_Images/TranslatedImage.jpg',14,140);
// res:=opencv.ROTATE('./test_images/test2.jpg','./Modified_Images/RotatedImage.jpg',-70);


/*Test case 3 for edge detection*/
// res:=opencv.DETECT_EDGE('./test_images/test3.jpg');

OUTPUT(res);
