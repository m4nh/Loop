import argparse
import glob
import os
import numpy as np
import cv2
from loop.loop import MotionEstimator, MotionEstimatorParameters

ap = argparse.ArgumentParser("Compute Rigid Motion")
ap.add_argument("--folder", required=True, help="Imges folder", type=str)
ap.add_argument("--orb_features", default=1000, help="ORB number of features", type=int)
args = vars(ap.parse_args())


# Motion Estimator
parameters = MotionEstimatorParameters()
parameters.algorithm = MotionEstimatorParameters.ALGORITHM_ORB
parameters.number_of_features = args['orb_features']
motionEstimator = MotionEstimator(parameters=parameters)

# Images
images = sorted(glob.glob(os.path.join(args['folder'], "*")))

test_points = np.array([
    [0, 0, 1],
    [100, 0, 1],
    [100, 100, 1],
    [0, 100, 1],
], int)

test_points += np.array([200, 200, 0])

current_index = 0
previous_image = cv2.imread(images[current_index])
last_image = previous_image
current_frame = np.eye(3)
frames = [np.eye(3)]
for index in range(1, len(images)):

    # current
    current_image = cv2.imread(images[index])

    H = motionEstimator.computeMotion(previous_image, current_image, debug=True)

    if H is None:
        print("Keyframe")

        H = motionEstimator.computeMotion(last_image, current_image, debug=True)
        current_frame = frames[index - 1]
        print("NEW H")
        previous_image = last_image

    current_H = np.matmul(current_frame, H)
    frames.append(current_H)
    print(current_H)
    # last
    last_image = current_image

    out = current_image.copy()

    current_points = np.matmul(current_H, test_points.T).T.astype(int)

    current_points = np.vstack((current_points, current_points[0, :]))
    for i in range(0, len(current_points)-1):
        p0 = current_points[i, :]
        p1 = current_points[i+1, :]
        cv2.line(out, (p0[0], p0[1]), (p1[0], p1[1]), (0, 0, 255), 2)

    cv2.imshow("out", out)
