import numpy as np
import cv2

import sys
import glob
import os
import argparse
import loop
from loop import InteractiveWindow

class_names = [
    'artifact_black',
    'artifact_metal',
    'artifact_orange',
    'artifact_white',
    'battery_black',
    'battery_green',
    'box_brown',
    'box_yellow',
    'clip',
    'glue',
    'pendrive',
    'screwdriver'
]
current_class_index = 0
polygons_map = {}
polygons_flow_map = {}
for name in class_names:
    polygons_map[name] = None
    polygons_flow_map[name] = []


current_image = None
current_polygon = None

orb = cv2.ORB_create(4000)


def storeData(filename, polygon, name, index, debug=True):
    labelfile = os.path.splitext(filename)[0] + ".npy"
    label_data = {}
    if os.path.exists(labelfile):
        label_data = np.load(labelfile).item()
    if name not in label_data:
        label_data[name] = {}
    label_data[name][index] = polygon
    if debug:
        print("Saving:", label_data)
    np.save(labelfile, label_data)


def computePolygonHomography(img1, img2, compute_affine=False, whole_mask=True, debug=False):

    if whole_mask:
        mask = np.ones((img1.shape[0], img1.shape[1]), dtype=np.uint8) * 255
    else:
        mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon1.astype(int)], (255))

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, mask)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if debug:
        out1 = img1.copy()
        out1 = cv2.drawKeypoints(img1, kp1, out1)
        out2 = img2.copy()
        out2 = cv2.drawKeypoints(img2, kp2, out2)
        out = np.vstack((out1, out2))
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w * 0.5), int(h * 0.5)),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imshow("mask", out)
        # cv2.waitKey(0)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    good = matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if compute_affine:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None
    else:
        M = cv2.estimateRigidTransform(src_pts, dst_pts, False)
        if M is None:
            return None
        M = np.vstack((M, np.array([0, 0, 1])))

    return M


def transformPoly(polygon, M):
    pts = np.float32(polygon).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    return dst.reshape((-1, 2))


class PointsBunch(object):

    def __init__(self, max_loop_clousure_dist=10):
        self.points = []
        self.max_loop_clousure_dist = max_loop_clousure_dist
        self.loop_closed = False

    def isValid(self):
        return len(self.points) > 0

    def checkLoopClosure(self, point, max_dist=10):
        if len(self.points) == 0:
            return False
        dist = np.linalg.norm(np.array(point) - self.points[0])
        if dist <= max_dist:
            return True
        return False

    def addPoint(self, point):
        loop_closed = self.checkLoopClosure(point)
        if not loop_closed:
            self.points.append(point)
        else:
            self.points.append(self.points[0].copy())
        return loop_closed

    def getPolygon(self):
        return np.array(self.points[0:-1])

    def draw(self, image, closed=False):
        points = self.points
        if closed:
            points = self.getPolygon()
        for i, p in enumerate(self.points):
            p1 = self.points[i]
            cv2.circle(image, tuple(p1), 4, (0, 0, 255), -1)
            if i < len(self.points) - 1:
                p2 = self.points[(i + 1)]
                cv2.circle(image, tuple(p2), 4, (0, 0, 255), -1)
                cv2.line(image, tuple(p1), tuple(p2), (255, 255, 0) if not closed else (
                    255, 255, 255), 1 if not closed else 3)


class ThreePointsBoxBunch(PointsBunch):
    def __init__(self):
        super(ThreePointsBoxBunch, self).__init__()

    def addPoint(self, point):
        if len(self.points) < 3:
            self.points.append(point)
        return len(self.points) == 3

    def getPolygon(self):
        p1 = np.array(self.points[0])
        p2 = np.array(self.points[1])
        p3 = np.array(self.points[2])
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        length = np.linalg.norm(p2 - p1)
        direction = (p2 - p1) / length
        orto = np.array([direction[1], -direction[0]])
        np0 = p1 + orto * d
        np1 = np0 + direction * length
        np2 = np1 - orto * 2 * d
        np3 = np2 - direction * length
        return np.array([np0, np1, np2, np3]).reshape((4, 2))

    def draw(self, image, closed=False):
        if not closed:
            return super(ThreePointsBoxBunch, self).draw(image, closed)
        else:
            poly = self.getPolygon()
            p1 = tuple(poly[0, :].astype(int))
            p2 = tuple(poly[1, :].astype(int))
            p3 = tuple(poly[2, :].astype(int))
            p4 = tuple(poly[3, :].astype(int))
            cv2.line(image, p1, p2, (0, 0, 255), 3)
            cv2.line(image, p1, p4, (0, 255, 0), 3)
            cv2.line(image, p2, p3, (255, 0, 0), 3)
            cv2.line(image, p3, p4, (255, 0, 0), 3)



# points_bunch = PointsBunch()
points_bunch = ThreePointsBoxBunch()


# def test(data):
#     global current_image, first_frame, window, current_polygon, points_bunch
#     print(data)

#     finish = points_bunch.addPoint(data[1])

#     current_image = first_frame.copy()
#     points_bunch.draw(current_image, finish)
#     if finish:
#         current_polygon = points_bunch.getPolygon()
#         print (current_polygon)


def mousedown(data):
    global current_image, first_frame, window, current_polygon, points_bunch
    if len(points_bunch.points) == 0:
        points_bunch.points = [tuple(np.array([0.0, 0.0]).astype(int))] * 2
        current_polygon = None
    elif len(points_bunch.points) == 2:
        points_bunch.points.append(tuple(np.array(data[1]).astype(int)))


def drawing(data):
    global current_image, first_frame, window, current_polygon, points_bunch, current_class_index, class_names
    if len(points_bunch.points) == 2:
        points_bunch.points[0] = data[1]
        points_bunch.points[1] = data[2]
        current_image = first_frame.copy()
        points_bunch.draw(current_image, False)
    if len(points_bunch.points) == 3:
        points_bunch.points[2] = data[2]
        current_image = first_frame.copy()
        points_bunch.draw(current_image, True)
        current_polygon = points_bunch.getPolygon()

        current_class_name = class_names[current_class_index]
        polygons_map[current_class_name] = points_bunch.getPolygon()


def reset(reset_current_class_polygon=False):
    global current_image, first_frame, window, current_polygon, points_bunch, current_class_index, class_names
    current_polygon = None
    current_image = first_frame.copy()
    points_bunch = ThreePointsBoxBunch()

    if reset_current_class_polygon:
        current_class_name = class_names[current_class_index]
        polygons_map[current_class_name] = None


ap = argparse.ArgumentParser("Tracker Labeler")
ap.add_argument("--folder", required=True, help="Imges folder")
ap.add_argument('--blind', dest='blind', action='store_true')
args = vars(ap.parse_args())

blind = args['blind']
print("Blind mode: ", blind)


# Input
folder = args['folder']
images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

window = InteractiveWindow("frame")


# First Frame Grab
window.registerCallback(mousedown, event=InteractiveWindow.EVENT_MOUSEDOWN)
window.registerCallback(drawing, event=InteractiveWindow.EVENT_DRAWING)

first_frame = cv2.imread(images[0])
current_image = first_frame.copy()
while True:

    for _, p in polygons_map.items():
        if p is not None:
            print("A " * 10, p.reshape((-1, 1, 2)).astype(int))
            cv2.polylines(current_image, [p.reshape((-1, 1, 2)).astype(int)], True, (0, 255, 255))

    #######################################
    # User Infoes
    #######################################
    current_class_name = class_names[current_class_index]
    class_polygon = polygons_map[current_class_name]
    if class_polygon is None:
        loop.ImageUtils.drawText(current_image, (10, 50), "Draw: {}".format(current_class_name), scale=1, padding=5, thickness=1, color=(255, 255, 255), bg=(0, 0, 255))
    else:
        loop.ImageUtils.drawText(current_image, (10, 50), "{} is OK!".format(current_class_name), scale=1, padding=5, thickness=1, color=(0, 0, 0), bg=(0, 255, 0))

    #######################################
    # Windows update
    #######################################
    c = window.showImg(current_image, 1)
    if c & 0xFF == ord('n'):
        # if points_bunch.isValid():
        #     polygons_map[current_class_name] = points_bunch.getPolygon()
        reset(False)
        current_class_index += 1
        current_class_index = current_class_index % len(class_names)
    if c & 0xFF == ord('q'):
        for name, p in polygons_map.items():
            if p is not None:
                polygons_flow_map[name].append(p)
        break
    if c & 0xFF == ord('c'):
        reset()
    if c & 0xFF == ord('s'):
        sys.exit(0)


current_index = 1
previous_frame = first_frame.copy()
last_frame = None
# last_polygon = current_polygon.copy()
while(current_index < len(images)):
    # Capture frame-by-frame
    current_image = images[current_index % len(images)]
    current_frame = cv2.imread(current_image)

    #######################################
    # Rigid transform
    #######################################
    M = computePolygonHomography(
        previous_frame,
        current_frame,
        compute_affine=False,
        whole_mask=True,
        debug=not blind
    )

    for name, p in polygons_map.items():
        if p is not None:

            new_poly = transformPoly(p, M)
            polygons_flow_map[name].append(new_poly)

    last_polygon = new_poly
    last_frame = current_frame.copy()

    if not blind:
        print("CURENT", current_polygon)
        print("NEW POLY", new_poly)

    # current_polygon = new_poly.copy()
    # previous_frame = current_frame.copy()

    output = current_frame.copy()

    for name, poly_list in polygons_flow_map.items():
        if len(poly_list) > 0:
            p = poly_list[-1]
            if p is not None:
                output = cv2.polylines(
                    output, [np.int32(p)], True, 255, 3, cv2.LINE_AA)

    loop.ImageUtils.drawText(output, (10, 50), "Frame: {}".format(current_index), scale=1, padding=5, thickness=1, color=(0, 0, 0), bg=(0, 255, 0))
    # output = cv2.polylines(
    #     output, [np.int32(new_poly)], True, 255, 3, cv2.LINE_AA)

    print("Current Frame: {}, Percentage: {:.2f}".format(
        current_index,
        float(current_index) / float(len(images)) * 100.0
    ))

    current_index += 1
    if current_index >= len(images):
        break
    # Display the resulting frame
    if not blind:
        cv2.imshow('frame', output)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

import pprint
pprint.pprint(polygons_flow_map)
# When everything done, release the capture
cv2.destroyAllWindows()
