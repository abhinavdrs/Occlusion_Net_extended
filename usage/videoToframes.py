"""
videoToframes.py takes a video as input and stores extracted framesin Occlusion_Net/Demo folder. 
the path to the videofile (including the name of video file itself) must be passed as -path .
"""

import cv2
import time
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Path to video')
    parser.add_argument('-path', type = dir_path)
    args = parser.parse_args()
    return args

def dir_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Video_Path:{path} is not valid")

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

def main():
    parsed_args = parse_arguments()
    input_loc = parsed_args.path
    output_loc = os.path.dirname(os.getcwd()) + '/demo'
    video_to_frames(input_loc, output_loc)


if __name__=="__main__":
    main()



