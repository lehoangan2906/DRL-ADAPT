#!/bin/sh

DIR="${1}"

# Start by checking if a directory "${1}" exists or not:
# If it does, print a message that it exists.
# If it does not, print an error message and create the directory.

if [ -d "$DIR"]; then
    ### Take action if $DIR exists ###
    echo "Directory ${DIR} exists"

else
    ###  Control will jump here if $DIR does NOT exists ###
    echo "Error: ${DIR} not found. Creating ${DIR}"
    mkdir ${DIR}
fi


Xvfb :1 -screen 0 1600x1200x16 &
export DISPLAY=:1.0
# This script sets up Xvfb to create a virtual display with resolution 1600x1200 and 16-bit color depth.
# The DISPLAY environment variable is then set to this virtual display.
# Ensuring that any application that requires a display can use this virtual display.


# Register gym:
cd ./drl_vo/src/turtlebot_gym/
pip3 install -e .
cd ../../
# This script navigates to the turtlebot_gym directory, which contains the custom Gym
# environment for the TurtleBot. It then installs the Gym environment using pip3.


# roslaunch training:
roslaunch drl_vo_nav drl_vo_nav_train.launch.py gui:="false" enable_opencv:="false" enable_consolve_output:="false" rviz:="false" log_dir:="${DIR}"
# This script use roslaunch to start the ROS based training process by launching the drl_vo_nav_train.launch file.
# The log_dir parameter is set to the directory provided before where logs and outputs from the training process will be stored.