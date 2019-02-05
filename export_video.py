import argparse
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser("Compute Rigid Motion")
ap.add_argument("--video", required=True, help="Video filename", type=str)
ap.add_argument("--output_folder", required=True, help="Output_folder", type=str)
ap.add_argument("--extension", default="jpg", help="Output extension", type=str)
args = vars(ap.parse_args())

# Ouptut folder
if not os.path.exists(args['output_folder']):
    os.mkdir(args['output_folder'])

# Video
cap = cv2.VideoCapture(args['video'])

pattern = "{:05d}_image." + args['extension']

saving_counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Our operations on the frame come here
        outputfile = os.path.join(args['output_folder'], pattern.format(saving_counter))
        cv2.imwrite(outputfile, frame)
        print("Saved -> ", outputfile)
        saving_counter += 1
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
