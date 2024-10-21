import cv2

def resize_video(input_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get the original video's width and height
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the new width and height for resizing
    new_width = 1920  # Full HD width
    new_height = 1080  # Full HD height

    # Create a VideoWriter object to save the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        # Resize the frame to Full HD
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Write the resized frame to the output video
        output_video.write(resized_frame)

    # Release the video objects
    video.release()
    output_video.release()

    print("Video resized and saved successfully!")

# Specify the input and output paths
input_path = "tmp.mp4"
output_path = "tmp_fullhd.mp4"

# Call the resize_video function
resize_video(input_path, output_path)