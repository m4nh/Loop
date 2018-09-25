import numpy as np
import os
import json
import sys
import cv2
import glob
import math
import random
import shutil
from dicttoxml import dicttoxml
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from xml.etree import ElementTree
from PIL import ImageFont, ImageDraw


class Score(object):

    def __init__(self):
        self.P = 0
        self.N = 0
        self.TP = 0
        self.FP = 0
        self.IOU = 0.0
        self.OIOU = 0.0
        self.ALIGN = 0.0

    def precision(self):
        if self.TP + self.FP == 0:
            return 0
        return self.TP / (self.TP + self.FP)

    def recall(self):
        if self.N == 0:
            return 0
        return self.TP / self.N

    def fscore(self):
        if self.precision() + self.recall() == 0:
            return 0
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())

    def avgIOU(self):
        if self.TP == 0:
            return 0.0
        return self.IOU / float(self.TP)

    def avgOIOU(self):
        if self.TP == 0:
            return 0.0
        return self.OIOU / float(self.TP)

    def avgALIGN(self):
        if self.TP == 0:
            return 0.0
        return self.ALIGN / float(self.TP)

    def getValueByName(self, field):
        if field.lower() == 'precision':
            return self.precision()
        if field.lower() == 'recall':
            return self.recall()
        if field.lower() == 'fscore':
            return self.fscore()
        if field.lower() == 'iou':
            return self.avgIOU()
        if field.lower() == 'oiou':
            return self.avgOIOU()
        if field.lower() == 'align':
            return self.avgALIGN()
        return 0.0

    def __repr__(self):
        return "Precision:{:.3f} Recall:{:.3f} FScore:{:.3f} avgIOU:{:.3f} avgOIOU:{:.3f} avgALIGN:{:.3f}".format(
            self.precision(),
            self.recall(),
            self.fscore(),
            self.avgIOU(),
            self.avgOIOU(),
            self.avgALIGN()
        )


class AngleDiscretization(object):

    def __init__(self):
        self.classes_size = 0


class AngleDiscretizationExp8(AngleDiscretization):

    def __init__(self):
        super(AngleDiscretizationExp8, self).__init__()
        self.classes_size = 7
        self.label_indices = [2.0, 30.0, 90.0, 180.0, 270.0, 330.0, 358.0]

    def getLabelByAngle(self, angle):
        label = 0
        for i in range(len(self.label_indices)):
            if np.abs(angle) > self.label_indices[i]:
                label = i + 1
        return label % len(self.label_indices)

    def getAngleAndArc(self, label):
        if label == 0:
            angle = 0.0
            arc = self.label_indices[0] * 2.0
        else:
            arc = self.label_indices[label] - self.label_indices[label - 1]
            angle = self.label_indices[label - 1] + arc * 0.5
        return angle, arc


class DARKENTUtils(object):

    @staticmethod
    def convertSimpleClassList(classlist, angle_discretization):

        if isinstance(angle_discretization, AngleDiscretization):
            newlist = []
            angle_classes_size = angle_discretization.classes_size
            for c in classlist:
                for i in range(angle_classes_size):
                    newlist.append('{}_{}'.format(c, i))
            return newlist
        else:
            newlist = []
            angle_classes_size = int(360.0 / angle_discretization) + 1
            for c in classlist:
                for i in range(angle_classes_size):
                    newlist.append('{}_{}'.format(c, i))
            return newlist

    @staticmethod
    def convertInstanceForAngleClassification2String(instance, angle_discretization):

        angle = instance.angle(deg=True, only_positive=True)

        if isinstance(angle_discretization, AngleDiscretization):
            angle_classes_size = angle_discretization.classes_size
            angle_class = angle_discretization.getLabelByAngle(angle)

            if angle_class > angle_classes_size:
                print("Angle error! ", angle, angle_classes_size, angle_class)
                sys.exit(0)

            new_angle_class = instance.label * angle_classes_size + angle_class
            bbox = instance.buildBoundingBox()
            return "{},{},{},{},{}".format(
                bbox['xmin'],
                bbox['ymin'],
                bbox['xmax'],
                bbox['ymax'],
                int(new_angle_class)
            )

        else:
            angle_classes_size = int(360.0 / angle_discretization) + 1
            angle_class = round(angle / angle_discretization)    # int()?

            if angle_class > angle_classes_size:
                print("Angle error! ", angle, angle_classes_size, angle_class)
                sys.exit(0)

            new_angle_class = instance.label * angle_classes_size + angle_class
            bbox = instance.buildBoundingBox()
            return "{},{},{},{},{}".format(
                bbox['xmin'],
                bbox['ymin'],
                bbox['xmax'],
                bbox['ymax'],
                int(new_angle_class)
            )

    @staticmethod
    def convertImageAnnotationsForAngleClassification2String(annotation, angle_discretization):
        image_path = annotation.getImagePath()
        row = "{}".format(image_path)

        for instance in annotation.getInstances():
            row += ' ' + DARKENTUtils.convertInstanceForAngleClassification2String(instance, angle_discretization)

        return row

    @staticmethod
    def exportAnnotationToDarknetDatasetForAngleClassification(output_file, annotations, angle_discretization, shuffle=True):
        f = open(output_file, 'w')

        if shuffle:
            random.shuffle(annotations)

        for index, annotation in enumerate(annotations):
            f.write(
                DARKENTUtils.convertImageAnnotationsForAngleClassification2String(annotation, angle_discretization)
            )
            if index < len(annotations) - 1:
                f.write('\n')
        f.close()


class VOCUtils(object):

    @staticmethod
    def convertInstanceForAngleClassification2DICT(instance, angle_discretization):

        angle = instance.angle(deg=True, only_positive=True)
        angle_classes_size = int(360.0 / angle_discretization) + 1
        angle_class = round(angle / angle_discretization)    # int()?

        if angle_class > angle_classes_size:
            print("Angle error! ", angle, angle_classes_size, angle_class)
            sys.exit(0)

        new_angle_class = instance.label * angle_classes_size + angle_class

        obj = {
            'bndbox': instance.buildBoundingBox(),
            'difficult': 0,
            'pose': 'Unspecified',
            'name': '{:.1f}'.format(new_angle_class),
            'truncated': 0
        }
        return obj

    @staticmethod
    def convertImageAnnotationsForAngleClassification2XML(annotation, angle_discretization):
        image_path = annotation.getImagePath()
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)

        if len(image.shape) == 3:
            h, w, d = image.shape
        else:
            h, w = image.shape
            d = 1

        data = {
            'annotation': {
                'path': image_path,
                'folder': 'images',
                'size': {
                    'width': w,
                    'height': h,
                    'depth': d
                },
                'source': {'database': 'Unknown'},
                'filename': image_name
            }
        }
        xmlstr = dicttoxml(data, root=False, custom_root='annotation', attr_type=False)
        document = ET.fromstring(xmlstr)
        for instance in annotation.getInstances():
            o = VOCUtils.convertInstanceForAngleClassification2DICT(instance, angle_discretization)
            ostr = dicttoxml({'object': o}, root=False, attr_type=False, custom_root='object')
            oel = ET.fromstring(ostr)
            document.append(oel)

        return document

    @staticmethod
    def saveXMLToFile(xml, filename, remove_header=True):
        # Export XML to file
        rough_string = ElementTree.tostring(xml, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="\t")
        if remove_header:
            header = pretty.split("\n")[0]
            pretty = pretty.replace(header, '')

        f = open(filename, 'w')
        f.write(pretty.strip())
        f.close()

    @staticmethod
    def exportAnnotationToVocDatasetForAngleClassification(dataset_folder, annotations, angle_discretization, annotations_subfolder='annotations', images_subfolder='images', zpadding=6):
        annotations_folder = os.path.join(dataset_folder, annotations_subfolder)
        images_folder = os.path.join(dataset_folder, images_subfolder)
        if not os.path.exists(annotations_folder):
            os.mkdir(annotations_folder)
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        index = 0
        for annotation in annotations:
            padded_index = str(index).zfill(zpadding)
            output_image_name = padded_index + '.' + annotation.getImageExtension()
            output_annotation_name = padded_index + '.xml'

            output_image_path = os.path.join(images_folder, output_image_name)
            output_annotation_path = os.path.join(annotations_folder, output_annotation_name)

            new_annotation = annotation.buildCopy(output_image_path=output_image_path)
            xml_document = VOCUtils.convertImageAnnotationsForAngleClassification2XML(new_annotation, angle_discretization)
            VOCUtils.saveXMLToFile(xml_document, output_annotation_path)

            index += 1
            perc = 100.0 * float(index) / float(len(annotations))
            print("Dataset build: {:.2f}".format(perc))


class ImageUtils(object):
    @staticmethod
    def sub_image(image, center, theta, width, height):
        """Extract a rectangle from the source image.

        image - source image
        center - (x,y) tuple for the centre point.
        theta - angle of rectangle.
        width, height - rectangle dimensions.
        """

        if 45 < theta <= 90:
            theta = theta - 90
            width, height = height, width

        theta *= math.pi / 180  # convert to rad
        v_x = (math.cos(theta), math.sin(theta))
        v_y = (-math.sin(theta), math.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
        mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

        return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def drawText(image, point, text, scale=1, padding=5, thickness=1, color=(33, 33, 33), bg=(59, 235, 255)):
        font_scale = scale
        font = cv2.FONT_HERSHEY_PLAIN
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((point[0], point[1]), (point[0] + text_width + padding*2, point[1] - text_height - padding*2))
        cv2.rectangle(image, box_coords[0], box_coords[1], bg, cv2.FILLED)
        cv2.putText(image, text, (point[0]+padding, point[1]-padding), font, fontScale=font_scale, color=color, thickness=thickness)


class DatasetManifest(object):

    def __init__(self, manifest_file):
        self.manifest_file = manifest_file
        _classmap = {}
        with open(manifest_file) as f:
            _classmap = json.load(f)
        self.classmap = {}
        self.classmap_inv = {}
        for l, name in _classmap.items():
            self.classmap[int(l)] = name
            self.classmap_inv[name] = int(l)

        self.models_ratios_path = os.path.join(os.path.dirname(self.manifest_file), 'models_ratios.txt')
        self.ratios = {}
        if os.path.exists(self.models_ratios_path):
            f = open(self.models_ratios_path, 'r')
            lines = f.readlines()
            for l in lines:
                classname = l.split(' ')[0]
                ratiostr = l.split(' ')[1]
                w, h = map(float, ratiostr.split(','))
                self.ratios[classname] = w / h

    def getPurgedList(self):
        k = sorted(self.classmap.keys())
        names = []
        for l in k:
            if l >= 0:
                names.append(self.getName(l))
        return names

    def getClassMap(self):
        return self.classmap

    def getLabels(self):
        return sorted(self.classmap.keys())

    def getName(self, l):
        if l in self.classmap:
            return self.classmap[l]
        return None

    def getRatio(self, name_or_label):
        name = name_or_label
        if isinstance(name_or_label, int):
            name = self.getLabel(name_or_label)
        if name in self.ratios:
            return self.ratios[name]
        return 1.0

    def getLabel(self, name):
        if name in self.classmap_inv:
            return self.classmap_inv[name]
        return -1


class InstanceSet(list):

    def __init__(self, instances):
        self.instances = instances

    def countLabel(self, label):
        counter = 0
        for i in self.instances:
            if i.label == label:
                counter += 1
        return counter

    def data(self):
        return self.instances


class Instance(object):
    MAX_IOU_CANVAS_SIZE = 4000

    def __init__(self, label=-1, points=np.array([]), score=1.0, dataset_manifest=None, unoriented_instance=False):
        self.label = label
        self.points = points
        self.score = score
        self.dataset_manifest = dataset_manifest
        if self.dataset_manifest is not None:
            self.name = dataset_manifest.getName(label)
        else:
            self.name = str(label)
        try:
            self.points = np.array(self.points).reshape((4, 2))
        except:
            print("4 Points Needed!")
            sys.exit(0)

        self.x_axis = self.points[1, :] - self.points[0, :]
        self.y_axis = self.points[3, :] - self.points[0, :]
        self.direction = self.x_axis / np.linalg.norm(self.x_axis)
        self.unoriented_instance = unoriented_instance

    def getDirection(self):
        return self.direction

    def angle(self, deg=False, only_positive=False):
        angle = np.arctan2(self.direction[1], self.direction[0])
        if only_positive:
            if angle < 0:
                angle = np.pi * 2 + angle
        if deg:
            return angle * 180.0 / np.pi
        else:
            return angle

    def ratio(self):
        return np.linalg.norm(self.x_axis) / np.linalg.norm(self.y_axis)

    def center(self):
        c = np.sum(self.points, axis=0)
        return c / 4.0

    def size(self):
        w = np.linalg.norm(self.x_axis)
        h = np.linalg.norm(self.y_axis)
        return np.array([w, h])

    def bboxSize(self):
        bbox = self.buildBoundingBox()
        return np.array([
            bbox['xmax'] - bbox['xmin'],
            bbox['ymax'] - bbox['ymin']
        ])

    def buildBoundingBox(self, asdict=True, dtype=int):
        min_p = np.min(self.points, axis=0).astype(dtype)
        max_p = np.max(self.points, axis=0).astype(dtype)
        if asdict:
            return {
                'xmin': min_p[0],
                'ymin': min_p[1],
                'xmax': max_p[0],
                'ymax': max_p[1],
            }
        else:
            return min_p, max_p

    def draw(self, image, fixed_color=None, custom_text=None):
        p0 = tuple(self.points[0, :].astype(int))
        p1 = tuple(self.points[1, :].astype(int))
        p2 = tuple(self.points[2, :].astype(int))
        p3 = tuple(self.points[3, :].astype(int))
        if fixed_color is not None:
            cv2.line(image, p0, p1, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p0, p3, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p1, p2, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p2, p3, fixed_color, 2, lineType=cv2.LINE_AA)
        else:
            cv2.line(image, p0, p1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(image, p0, p3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(image, p1, p2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(image, p2, p3, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(image, tuple(self.center().astype(int)), 2, (255, 255, 255), -1)
        angle_deg = self.angle() * 180.0 / np.pi
        title = "{}_{:.2f}".format(
            self.name,
            angle_deg
        )
        if custom_text is not None:
            title = custom_text
        cv2.putText(image, title, p0, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def draw2(self, image, fixed_color=None, print_label=False, custom_text=None):
        p0 = tuple(self.points[0, :].astype(int))
        p1 = tuple(self.points[1, :].astype(int))
        p2 = tuple(self.points[2, :].astype(int))
        p3 = tuple(self.points[3, :].astype(int))
        min_point = self.points[np.argmax(self.points[:, 1], axis=0), :].copy()
        min_point[0] = self.center()[0]
        min_point[1] += 20
        min_point = tuple(min_point.astype(int))

        if fixed_color is not None:
            cv2.line(image, p0, p1, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p0, p3, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p1, p2, fixed_color, 2, lineType=cv2.LINE_AA)
            cv2.line(image, p2, p3, fixed_color, 2, lineType=cv2.LINE_AA)
        else:
            cv2.line(image, p1, p2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(image, p2, p3, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.arrowedLine(image, p0, p1, (0, 0, 255), 2, line_type=cv2.LINE_AA)
            cv2.arrowedLine(image, p0, p3, (0, 255, 0), 2, line_type=cv2.LINE_AA)

        cv2.circle(image, tuple(self.center().astype(int)), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        angle_deg = self.angle() * 180.0 / np.pi

        title = self.dataset_manifest.getName(self.label)
        if print_label:
            title = str(self.label)
        if custom_text is not None:
            title = custom_text

        if custom_text is None:
            custom_text = ''
        if len(custom_text) > 0:
            ImageUtils.drawText(image, min_point, title)

    def toString(self, with_score=False):
        l = [self.label] + self.points.astype(int).ravel().tolist()
        if with_score:
            l = l + ["{:.2f}".format(self.score)]
        return ','.join(map(str, l))

    def computeIOU(self, other):
        stacked = np.vstack((self.points, other.points))
        p_max = np.max(stacked, axis=0).astype(float)
        p_max = (p_max * 1.2).astype(int)
        p_max = np.clip(p_max, 0, Instance.MAX_IOU_CANVAS_SIZE)

        m1 = np.zeros((p_max[1], p_max[0]), dtype=np.uint8)
        m2 = np.zeros((p_max[1], p_max[0]), dtype=np.uint8)
        # print(np.int32(self.points))
        # print(np.int32(other.points))
        cv2.fillPoly(m1, [np.int32(self.points)], (255))
        cv2.fillPoly(m2, [np.int32(other.points)], (255))
        # cv2.imshow("m1", m1)
        # cv2.imshow("m2", m2)
        intersection_mask = cv2.bitwise_and(m1, m2)
        union_mask = cv2.bitwise_or(m1, m2)

        intersection = np.count_nonzero(intersection_mask)
        union = np.count_nonzero(union_mask)
        iou = float(intersection) / float(union)
        return iou

    def computeOIOU(self, other):
        iou = self.computeIOU(other)
        sim = np.dot(self.getDirection(), other.getDirection())
        return max(0, sim * iou)

    def computeALIGN(self, other):
        return max(0, np.dot(self.getDirection(), other.getDirection()))

    @staticmethod
    def fromString(s):
        try:
            chunks = s.split(',')
            if len(chunks) >= 9:
                label = int(chunks[0])
                points = np.array(chunks[1:9]).astype(float).reshape((4, 2))
                score = 1.0
                if len(chunks) > 9:
                    score = float(chunks[9])
                return Instance(label=label, points=points, score=score)
            elif len(chunks) == 5:
                label = int(chunks[4])
                min_p = np.array(chunks[0:2]).astype(float)
                max_p = np.array(chunks[2:4]).astype(float)
                size = max_p - min_p
                w = size[0]
                h = size[1]
                points = np.array([
                    [min_p],
                    [min_p + np.array([w, 0])],
                    [min_p + np.array([w, h])],
                    [min_p + np.array([0, h])]
                ])
                return Instance(label=label, points=points, unoriented_instance=True)
            elif len(chunks) == 6:
                label = chunks[0]
                min_p = np.array([chunks[2], chunks[1]]).astype(float)
                max_p = np.array([chunks[4], chunks[3]]).astype(float)
                score = float(chunks[5])
                size = max_p - min_p
                w = size[0]
                h = size[1]
                points = np.array([
                    [min_p],
                    [min_p + np.array([w, 0])],
                    [min_p + np.array([w, h])],
                    [min_p + np.array([0, h])]
                ])
                return Instance(label=label, points=points, score=score, unoriented_instance=True)

        except Exception as e:
            print("Error: ", e)
            sys.exit(0)

    def findMostSimilarInstance(self, instances, consider_orientation=False, th=0.5):
        oious = []
        atleastone = False
        correct_instances = []
        for i in instances:
            if self.label == i.label:
                correct_instances.append(i)
        if len(correct_instances) == 0:
            return None

        for i in correct_instances:
            if consider_orientation:
                oiou = self.computeOIOU(i)
            else:
                oiou = self.computeIOU(i)
            if oiou >= th:
                oious.append(oiou)
                atleastone = True
            else:
                oious.append(0.0)
        if not atleastone:
            return None

        oious = np.array(oious)
        maxi = np.argmax(oious)
        return correct_instances[maxi]

    @staticmethod
    def convertInstancesToRowString(image_path, instances):
        row = str(image_path)
        for i in instances:
            row += ' ' + i.toString()
        return row

    @staticmethod
    def parseRowString(row, relative_path=None):
        chunks = row.split(' ')
        image_path = ''
        instances = []
        try:
            image_path = chunks[0]
            if relative_path is not None:
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.path.dirname(relative_path), image_path)
                    print("Is relative path", image_path)
            if len(chunks) > 1:
                for i in range(1, len(chunks)):
                    instances.append(Instance.fromString(chunks[i]))
        except Exception as e:
            print("Error: ", e)
            pass
        return image_path.replace('\n', ''), instances

    @staticmethod
    def generateUnrotatedInstance(label, bbox, score=1.0):
        w = np.abs(bbox['xmax'] - bbox['xmin'])
        h = np.abs(bbox['ymax'] - bbox['ymin'])
        p_min = np.array([bbox['xmin'], bbox['ymin']]).astype(float)
        points = np.array([
            p_min,
            p_min + np.array([w, 0]),
            p_min + np.array([w, h]),
            p_min + np.array([0, h])
        ])
        return Instance(label, points, score=score)

    @staticmethod
    def generateRotatedInstance(label, bbox, angle, ratio, score=1.0):
        p_min = np.array([bbox['xmin'], bbox['ymin']]).astype(float)
        p_max = np.array([bbox['xmax'], bbox['ymax']]).astype(float)
        center = (p_min + p_max) / 2.0
        diag = np.linalg.norm(p_min - p_max)
        start_size = int(diag * 0.4)

        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])

        w = 0.5 * (start_size)
        h = 0.5 * (start_size / ratio)
        points = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ])
        points_nominal = np.dot(rot, points.T).T
        points = points_nominal + center

        min_dist = diag * 10
        min_index = -1
        for i in range(4):
            diff = np.abs(p_min[0] - points[i][0])
            if diff < min_dist:
                min_dist = diff
                min_index = i

        x_big = np.abs(p_min[0] - center[0])
        x_small = np.abs(points[min_index][0] - center[0])
        scale = x_big / x_small
        points_nominal = points_nominal * scale
        points = points_nominal + center
        return Instance(label, points, score=score)

    def __repr__(self):
        return "Instance({},{:.2f},{:.2f})".format(self.label, self.angle(), self.score)


class ImageAnnotation(object):

    def __init__(self, image_path='', annotation_path='', dataset_manifest=None):
        if dataset_manifest is None:
            print("Dataset Manifest not present!")
            sys.exit(0)

        self.setDatasetManifest(dataset_manifest)
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.instances = []

        if len(image_path) == 0 and len(annotation_path) == 0:
            return

        extension = os.path.splitext(annotation_path)[1]

        if 'npy' in extension:
            data = np.load(annotation_path).item()
            for label_name, instances in data.items():
                label = self.dataset_manifest.getLabel(label_name)
                for _, instance in instances.items():
                    inst = Instance(label=label, points=instance, dataset_manifest=dataset_manifest)
                    self.instances.append(inst)

    def getImageExtension(self):
        if len(self.image_path) > 0:
            ext = os.path.splitext(self.image_path)[1]
            return ext.replace('.', '')
        return ''

    def getInstances(self):
        return self.instances

    def getAnnotationPath(self):
        return self.annotation_path

    def getImagePath(self):
        return self.image_path

    def getBasename(self):
        return os.path.basename(self.getImagePath())

    def setDatasetManifest(self, dataset_manifest):
        self.dataset_manifest = dataset_manifest
        if isinstance(self.dataset_manifest, str):
            self.dataset_manifest = DatasetManifest(self.dataset_manifest)

    def loadImage(self):
        return cv2.imread(self.image_path)

    def draw(self, image):
        for a in self.instances:
            a.draw(image)

    def buildCopy(self, output_image_path='', output_annotation_path='', do_physical_copy=True):
        new_annotation = ImageAnnotation('', '', self.dataset_manifest)

        # Copy (if) the image
        if len(output_image_path) > 0:
            new_annotation.image_path = output_image_path
            if do_physical_copy:
                shutil.copy(self.image_path, output_image_path)
        else:
            new_annotation.image_path = self.image_path

        # Copy (if) the annotation
        if len(output_annotation_path) > 0:
            new_annotation.annotation_path = output_annotation_path
            if do_physical_copy:
                shutil.copy(self.annotation_path, output_annotation_path)
        else:
            new_annotation.annotation_path = self.annotation_path

        new_annotation.instances = self.instances.copy()

        return new_annotation


class DatasetOptions(object):

    def __init__(self):
        self.image_extension = 'jpg'
        self.annotation_extension = 'npy'
        self.auto_load_images = False


class DatasetScene(object):

    def __init__(self, dataset_scene_path='', dataset_manifest=None, dataset_options=DatasetOptions()):
        self.path = dataset_scene_path
        self.dataset_options = dataset_options
        self.dataset_manifest = dataset_manifest
        self.images_path = os.path.join(self.path, "images")
        self.annotations_path = os.path.join(self.path, "flowbel_annotations")
        self.name = os.path.basename(dataset_scene_path)
        self.image_annotations = []

        self.ready = False

        if dataset_options.auto_load_images:
            self.loadAnnotations()

    def getImageAnnotations(self):
        self.loadAnnotations()
        return self.image_annotations

    def loadAnnotations(self):
        if not self.ready:
            print("Loading scene: {}".format(self.name))
            try:
                images = sorted(glob.glob(os.path.join(self.images_path, "*." + self.dataset_options.image_extension)))
                annots = sorted(glob.glob(os.path.join(self.annotations_path, "*." + self.dataset_options.annotation_extension)))
                for i, img in enumerate(images):
                    self.image_annotations.append(
                        ImageAnnotation(
                            image_path=img,
                            annotation_path=annots[i],
                            dataset_manifest=self.dataset_manifest
                        )
                    )
                self.ready = True
            except Exception as e:
                print("Images and annotations mismatch!", e)
                sys.exit(0)

    def size(self):
        self.loadAnnotations()
        return len(self.image_annotations)

    def getImageAnnotation(self, index):
        self.loadAnnotations()
        if index < len(self.image_annotations) and index >= 0:
            return self.image_annotations[index]
        if index == -1:
            return random.choice(self.image_annotations)
        return None

    def extractSample(self, target_name, target_angle=0.0, padding=10):
        min_angle = 1180.0
        min_inst = None
        min_annot = None
        for i in range(self.size()):
            annotation = self.getImageAnnotation(i)
            found = False
            if annotation is not None:
                for inst in annotation.instances:
                    if inst.name == target_name:
                        dist = np.abs((inst.angle() - target_angle))
                        if dist < min_angle:
                            min_inst = inst
                            min_annot = annotation
                            min_angle = dist

        if min_annot is None:
            return None, None, None
        image = min_annot.loadImage()
        size = min_inst.size() + padding
        sub = ImageUtils.sub_image(image, min_inst.center(), min_inst.angle(deg=True), int(size[0]), int(size[1]))
        return sub, min_inst, min_annot


class Dataset(object):

    def __init__(self, dataset_path='', dataset_options=DatasetOptions()):
        self.dataset_options = dataset_options
        self.dataset_path = dataset_path
        self.classmap_path = os.path.join(self.dataset_path, "class_map.txt")
        if not os.path.exists(self.classmap_path):
            print("No class map manifest found!")
            sys.exit(0)
        self.dataset_manifest = DatasetManifest(self.classmap_path)
        self.scenes = {}
        self.loadScenes(dataset_path)

    def loadScenes(self, path):
        self.scenes = {}
        items = sorted(glob.glob(os.path.join(path, "*")))
        for i in items:
            if os.path.isdir(i):
                scene = DatasetScene(i, dataset_manifest=self.dataset_manifest, dataset_options=self.dataset_options)
                self.scenes[scene.name] = scene

    def getSceneByName(self, name):
        if name in self.scenes:
            return self.scenes[name]
        return None

    def getDatasetManifest(self):
        return self.dataset_manifest

    def generateInstanceFromClassification(self, box_data, angle_discretization, estimate_rotated_box=True):
        chunks = box_data.split(',')
        extended_label = chunks[0]
        class_name = extended_label.rsplit('_', 1)[0]
        angle_class = int(extended_label.rsplit('_', 1)[1])
        label = self.dataset_manifest.getLabel(class_name)
        angle = float(angle_class * angle_discretization) * np.pi / 180.0
        data = chunks[1:]
        ymin = float(data[0])
        xmin = float(data[1])
        ymax = float(data[2])
        xmax = float(data[3])
        score = 0.0
        if len(data) > 4:
            score = float(data[4])
        bbox = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
        }
        if estimate_rotated_box:
            return Instance.generateRotatedInstance(label, bbox, angle, self.dataset_manifest.getRatio(class_name), score=score)
        else:
            return Instance.generateUnrotatedInstance(label, bbox, score=score)

    @staticmethod
    def loadDataFromFile(filename, confidence_th=0.0):
        f = open(filename, 'r')
        lines = f.readlines()
        entries_map = {}
        for l in lines:
            image_path, instances = Instance.parseRowString(l)

            confident_instances = []
            for i in instances:
                if i.score >= confidence_th:
                    confident_instances.append(i)

            entries_map[image_path] = {
                'image_path': image_path,
                'instances': confident_instances
            }
        return entries_map


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class InteractiveWindowKeys(object):
    KEY_ARROW_LEFT = 81
    KEY_ARROW_RIGHT = 83


class InteractiveWindow(object):
    EVENT_DRAWING = "EVENT_DRAWING"
    EVENT_CLEARING = "EVENT_CLEARING"
    EVENT_MOUSEDOWN = "EVENT_MOUSEDOWN"
    EVENT_MOUSEUP = "EVENT_MOUSEUP"
    EVENT_MOUSEMOVE = "EVENT_MOUSEMOVE"
    EVENT_QUIT = "EVENT_QUIT"
    EVENT_KEYDOWN = "EVENT_KEYDOWN"

    def __init__(self, name, autoexit=False):
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.mouseCallback)

        self.drawing = {
            'status': False,
            'start': None
        }
        self.drawing_start_position = None
        self.clearing = False
        self.callbacks = []
        self.callbacks_map = {}
        self.autoexit = autoexit

    def startDrawing(self, position, status=True):
        if self.drawing['status'] != status and self.drawing['status'] is False:
            self.drawing['start'] = np.array(position)
        self.drawing['status'] = status

    def mouseCallback(self, event, x, y, flags, param):
        point = np.array([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.startDrawing((x, y))
            self.fireEvent(InteractiveWindow.EVENT_MOUSEDOWN, (0, point))

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing['status'] is True:
                self.fireEvent(InteractiveWindow.EVENT_DRAWING,
                               (0, self.drawing['start'],
                                point, None))
            elif self.clearing is True:
                self.fireEvent(InteractiveWindow.EVENT_CLEARING, (1, point))
            else:
                self.fireEvent(InteractiveWindow.EVENT_MOUSEMOVE, (1, point))

        if event == cv2.EVENT_LBUTTONUP:
            if self.drawing['status'] is True:
                self.fireEvent(InteractiveWindow.EVENT_DRAWING,
                               (0, self.drawing['start'],
                                point, point))
                self.startDrawing((x, y), False)
            self.fireEvent(InteractiveWindow.EVENT_MOUSEUP, (0, point))

        if event == cv2.EVENT_MBUTTONDOWN:
            self.clearing = True
            self.fireEvent(InteractiveWindow.EVENT_MOUSEDOWN, (2, point))

        if event == cv2.EVENT_MBUTTONUP:
            self.clearing = False
            self.fireEvent(InteractiveWindow.EVENT_MOUSEUP, (2, point))

    def showImg(self, img=np.zeros((500, 500)), time=0, disable_keys=False):
        # res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(self.name, img)
        if time >= 0:
            c = cv2.waitKey(time)
            if disable_keys:
                return
            # print "CH", c
            c = c & 255
            if c != 255:
                self.fireEvent(InteractiveWindow.EVENT_KEYDOWN, (chr(c), c))
                if c == 113:
                    self.fireEvent(InteractiveWindow.EVENT_QUIT, None)
                    if self.autoexit:
                        sys.exit(0)
            return c
        return -1

    def fireEvent(self, evt, data):
        for c in self.callbacks:
            c(evt, data)
        for event, cbs in self.callbacks_map.items():
            if event == evt:
                for cb in cbs:
                    cb(data)

    def registerCallback(self, callback, event=None):
        if event is None:
            self.callbacks.append(callback)
        else:
            if event not in self.callbacks_map:
                self.callbacks_map[event] = []
            self.callbacks_map[event].append(callback)

    def removeCallback(self, callback, event=None):
        if event is None:
            self.callbacks.remove(callback)
        else:
            if event not in self.callbacks_map:
                self.callbacks_map[event].remove(callback)
