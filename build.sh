#!/bin/bash

HPCC_CODE_DIR=$1

if [ -z "${HPCC_CODE_DIR}" ]; then
     echo "Usage: $0 <HPCC-Platform-Dir>"
     exit
fi

if [ ! -d "${HPCC_CODE_DIR}" ]; then
    echo "${HPCC_CODE_DIR} does not exist"
    exit
fi

g++ -c -std=c++11 -fPIC -pthread \
    -I${HPCC_CODE_DIR}/system/include \
    -I${HPCC_CODE_DIR}/system/jlib \
    -I${HPCC_CODE_DIR}/rtl/eclrtl \
    -I${HPCC_CODE_DIR}/rtl/include \
    -o opencvlib.o \
    opencvlib.cpp

g++ -c -std=c++11 -fPIC -pthread \
    -o ./License_Plate_Files/PossiblePlate/PossiblePlate.o \
    ./License_Plate_Files/PossiblePlate/PossiblePlate.cpp
 
g++ -c -std=c++11 -fPIC -pthread \
    -o ./License_Plate_Files/PossibleChar/PossibleChar.o \
    ./License_Plate_Files/PossibleChar/PossibleChar.cpp

g++ -c -std=c++11 -fPIC -pthread \
    -o ./License_Plate_Files/Preprocess/Preprocess.o \
    ./License_Plate_Files/Preprocess/Preprocess.cpp

g++ -c -std=c++11 -fPIC -pthread \
    -o ./License_Plate_Files/DetectPlates/DetectPlates.o \
    ./License_Plate_Files/DetectPlates/DetectPlates.cpp

g++ -c -std=c++11 -fPIC -pthread \
    -o ./License_Plate_Files/DetectChars/DetectChars.o \
    ./License_Plate_Files/DetectChars/DetectChars.cpp



g++ ./License_Plate_Files/PossibleChar/PossibleChar.o ./License_Plate_Files/PossiblePlate/PossiblePlate.o ./License_Plate_Files/Preprocess/Preprocess.o ./License_Plate_Files/DetectPlates/DetectPlates.o ./License_Plate_Files/DetectChars/DetectChars.o opencvlib.o -shared -pthread -lcurl `pkg-config opencv --cflags --libs` -o libopencv.so

echo "Built.  Use ./install.sh to install the library and/or interface file"
