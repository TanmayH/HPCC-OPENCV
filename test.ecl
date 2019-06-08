IMPORT opencv.opencv;

/*Test case 1 for license plate detection*/
res := opencv.loadKNNDataAndTrainKNN();
// OUTPUT(res);
IF (res=false,OUTPUT('error: error: KNN traning was not successful'),OUTPUT(opencv.GETLICENSEPLATE('./test_images/test2.jpg')));


/*Test Case 2 for individual functions*/
// res:=opencv.GETBLUR('./test_images/test2.jpg');
// res:=opencv.GRAYSCALE('./test_images/test2.jpg');
// res:=opencv.THRESH_HOLD('./test_images/test2.jpg',100);
// res:=opencv.RESIZE('./test_images/test2.jpg',0.25,0.75);
// res:=opencv.TRANSLATE('./test_images/test2.jpg',4,4);
// res:=opencv.ROTATE('./test_images/test2.jpg',70,1,10,10);
// OUTPUT(res);

/*Test case 3 for feature detection*/
// res:=opencv.MATCH_FEATURES('./test_images/test2.jpg','./test_images/test2.jpg');
// OUTPUT(res);

/*Test case 4 for edge detection*/
// res:=opencv.DETECT_EDGE('./test_images/test2.jpg');
// OUTPUT(res);