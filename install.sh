#!/bin/bash

if [ ! -e "opencv.ecllib" ]; then
	echo "opencv.ecllib does not exist"
	exit
fi

# Copy shared library to cluster plugins directory
if [ -d "/opt/HPCCSystems/plugins" ]; then
	if [ -e "libopencv.so" ]; then
		sudo cp libopencv.so /opt/HPCCSystems/plugins
	fi
	sudo cp opencv.ecllib /opt/HPCCSystems/plugins
fi

# Copy .ecllib file to every clienttools installation
sudo find /opt/HPCCSystems -name clienttools -exec cp opencv.ecllib {}/plugins \;
