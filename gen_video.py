import cv2
import time
import numpy as np
import os
from testface import video2anime

def video_to_frames(video_in, video_out, tface):

    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(video_in)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out,fourcc, fps, (width, height) )

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        cv2.imwrite("bf.jpg", frame)
        
        frame = cv2.resize(frame, (1920, 1920))
        frame = tface.style_transfer(frame)
        frame = cv2.resize(frame, (width, height))

        cv2.imwrite("af.jpg", frame)
        out.write(frame)
        count = count + 1
        print ("Frame: ", count)
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            out.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":
    tface = video2anime()
    video_in = "final_audio.mp4"
    video_out = "out.mp4"
    video_to_frames(video_in, video_out, tface)
