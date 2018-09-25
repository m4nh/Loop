# LOOP

## Datasets

### Loop Dataset 2018

Download Link: (https://vision.deis.unibo.it:5001/sharing/Ks5d1Ipi8)

### Loop Dataset Synthetic 2018

Download Link: (https://vision.deis.unibo.it:5001/sharing/PXGfEnbrj)

### Dataset Structure

The tree structure of each dataset is the following (bold strings represent fodlers):

* **scan_01**
  * **images**
    * 0000000.jpg
    * 0000001.jpg
    * ...
* **scan_02**
* **scan_XX**
* scan_01.txt
* scan_02.txt
* scan_XX.txt
* class_map.txt
* models_ratios.txt

So, each dataset containts a list of "scans" folders, each of which with an **images** subfolder containing row pictures. For each **Scan** folder there is a *scan_XX.txt* file with Ground Truth labels. The labels file is a set of rows like the following:

```
image_path <INSTANCE> <INSTANCE>
```

where each istance is:

```
label_id,p0_x,p0_y,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y
```
an example, with two instances, could be:

```
scan_01/images/0000000.jpg 0,369,153,310,136,325,83,384,100 1,307,325,365,371,342,400,284,354
```

## Tools

### Draw Predictions

The draw prediction tool is an example on how to read Labels Files (like the ones described above) just for visualization puroposes (the code may be extened for custom data conversion e.g.).

```
python draw_predictions.py --dataset_path $DATASET_FOLDER --labels_file $LABELS_FILE
```
Where ```$DATASET_FOLDER``` is an Env variable containing the base folder of one of the two dataset described in the first section. While, ```$LABELS_FILE``` is the full path of one of the labels file (like *scan_01.txt*). The tool will show all the images present in the label files drawing all labels associated with it.

### Labeling Demo

The labeling tool shows an example of automated labeling as shown in the supplementray material video:

```
python loop_labeler.py --folder $IMAGES_FOLDER
```

where ```$IMAGES_FOLDER``` is a folder containing a sequence of images (like described in the paper). You can use one of the **scan_XX/images** folder described above as a first test.
When the tool starts will show a ToolTip guiding the user to draw a specific object (drag&drop with mouse). After the manual labeling stage press ```q``` to start the automated labeling procedure for the following frames.

If you need the full code for the labeling pipeline please contact me at: **d.degregorio AT unibo.it**



