import numpy as np
import cv2
from loop import Instance
import loop
import argparse
import os
import time
import random

ap = argparse.ArgumentParser("Loop Draw Predictions")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--labels_file", required=True, help="Result file")
ap.add_argument("--min_th", default=0.5, help="Min TH", type=float)
args = vars(ap.parse_args())

# Creates a DATASET MODEL
dataset = loop.Dataset(dataset_path=args['dataset_path'])

# Opens LABELS file
f = open(args['labels_file'], 'r')
rows = f.readlines()

# Iterate labels rows
for counter, r in enumerate(rows):

    image_path, instances = Instance.parseRowString(r, args['labels_file'])
    if image_path is None or len(image_path) == 0:
        continue
    image = cv2.imread(image_path)

    for i in instances:
        i.dataset_manifest = dataset.dataset_manifest

        if i.score < args['min_th']:
            continue

        if i.unoriented_instance:
            i.draw2(image, fixed_color=(59, 235, 255), custom_text='')
        else:
            i.draw2(image, custom_text=dataset.dataset_manifest.getName(i.label))

    cv2.imshow("output", image)
    cv2.waitKey(0)
