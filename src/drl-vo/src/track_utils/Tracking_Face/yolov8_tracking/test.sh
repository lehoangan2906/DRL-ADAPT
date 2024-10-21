# for filename in /media/v100/DATA4/huynq/tmp/*.mp4; do
#     python track_face.py --device 3 --source "$filename"
# done
# python track_face.py --device 0 --source rtsp://admin:123456a%40@172.21.111.111:554/main &
# sleep 5 
# python testDBscan.py
#!/bin/bash

# Delay in seconds before running the program (e.g., 1 hour = 3600 seconds)
#delay = 15 


run_program() {
    # Change to the directory where your program is located

    # Run your program
    echo "TTTTTTTTTTTTTTTTTTTTTTTTT"
    python track_face.py --device 3 --source rtsp://admin:123456a%40@172.21.111.111
}

# Sleep for the specified delay
sleep 15h #$delay

# Run the program after the delay
run_program