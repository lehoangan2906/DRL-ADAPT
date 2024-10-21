- The folder `drl_vo` contains two sub-folders:
    + `src`: which contains all the source packages of the project.
    + `launch`: which contains all the launch files of the project.

- In the same level as the `drl_vo` folder, there are 3 .sh files:
    + `run_drl_vo_policy_training_desktop.sh`: Used to start the training process on a desktop with GUI (RVIZ). 
        + It initializes a virtual display.
        + Registers the custom Gym environment.
        + Launches the training process.
        
    + `run_drl_vo_policy_training_server.sh`: Used to start the training process on a server without GUI. It operates similarly to the desktop script but configured the environment without graphical output. 

    + `run_drl_vo_navigation_demo.sh`: This script is used to start the navigation demo.
        + It calls the `drl_vo_nav.launch` file, which run the trained DRL model in inference mode to navigate in a simulated or real environment.