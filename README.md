# HPCC-OPENCV
OpenCV Plugin for HPCC Platform  
  
    
    To run:  
    1). Setup opencv on the local cluster. Ensure that the libraries are present at the pkg-config location  
    2). Clone the HPCC Platform from their github link: https://github.com/hpcc-systems/HPCC-Platform  
    3). cd into the HPCC_OPENCV directory  
    4). Run the build file using : ./build.sh <Location of cloned HPCC Platform>  
    5). Run the install file using : ./install.sh  
    6). Compile the test.ecl file using : ecl run test.ecl --target=thor --server=<IP Address of server>  
    
    Note:  
    1).Make sure permissions for running build.sh and install.sh file are set if necessary  
    2).To run the various types of functions comment and uncomment the required portions from the test.ecl file
