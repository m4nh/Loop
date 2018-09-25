## Loop Dataset 2018

Download Link: (https://vision.deis.unibo.it:5001/sharing/Z4vAqomGj)

## Loop Dataset Synthetic 2018

Download Link: (https://vision.deis.unibo.it:5001/sharing/NA7DLnq14)

## Dataset Structure

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


