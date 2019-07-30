IMPORT opencv.opencv;

/*Record Formats*/

/*Image Record Format*/
imageRecord := RECORD 
  STRING filename;
  DATA   image;    //first 4 bytes contain the length of the image data
  UNSIGNED8  RecPos{virtual(fileposition)};
END;

/*Output LicensePlate file record format*/
numPlateRec := RECORD
    STRING fileName;
    STRING numPlate;
END;

/*Output NewImage file record format*/
newImageRec := RECORD
    STRING fileName;
    DATA newImage;
END;

/*Input image data*/
imageData := DATASET('~images::imagedb',imageRecord,FLAT);

/*---------------------------------------------------------------------------------------------*/
/*Test case 1 for license plate detection*/
/*Get the number plate and store it in numPlateRec*/
numPlateRec getNumPlate(imageData L, INTEGER C) := TRANSFORM
                SELF.fileName := L.filename;
                SELF.numPlate := opencv.GETLICENSEPLATE(L.image);
END;

/*Read each record in the image file and call the getNumPlate function*/
numPlateRecs := PROJECT(imageData,getNumPlate(LEFT,COUNTER),LOCAL);


/*Write the record set with numplates to HPCC*/
OUTPUT(numPlateRecs,,'~images::licensePlates', THOR, OVERWRITE);
output(numPlateRecs);


/*---------------------------------------------------------------------------------------------*/
/*Test case 2 for image to image conversion*/
/*Get the modified image data and store it in newImagerec*/
// newImageRec getNewImage(imageData L, INTEGER C) := TRANSFORM
//                 SELF.fileName := L.filename;
//                 // Possible individual functions
//                 // SELF.newImage := opencv.GAUSSBLUR(L.image, 21);
//                 // SELF.newImage := opencv.GRAYSCALE(L.image);
//                 // SELF.newImage := opencv.THRESH_HOLD(L.image, 38);
//                 // SELF.newImage := opencv.RESIZE(L.image, 0.15,0.95);
//                 // SELF.newImage := opencv.TRANSLATE(L.image, 14,140);
//                 // SELF.newImage := opencv.ROTATE(L.image, -70);
// END;

// /*Read each record in the image file and call the getNewImage function*/
// newImageRecs := PROJECT(imageData,getNewImage(LEFT,COUNTER),LOCAL);

// /*Write the record set with new image data to HPCC*/
// OUTPUT(newImageRecs,,'~images::modifiedImageDb', THOR, OVERWRITE);
// output(newImageRecs);
/*---------------------------------------------------------------------------------------------*/

/*Test case 3 for edge detection*/
/*Get the modified image data and store it in newImagerec*/
// newImageRec getEdges(imageData L, INTEGER C) := TRANSFORM
//                 SELF.fileName := L.filename;
//                 SELF.newImage := opencv.DETECT_EDGE(L.image, 0);
// END;

// /*Read each record in the image file and call the getEdges function*/
// newImageRecs := PROJECT(imageData,getEdges(LEFT,COUNTER),LOCAL);

// /*Write the record set with new image data to HPCC*/
// OUTPUT(newImageRecs,,'~images::newImageEdges', THOR, OVERWRITE);
// output(newImageRecs);
/*---------------------------------------------------------------------------------------------*/
