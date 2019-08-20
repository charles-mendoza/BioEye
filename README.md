# Face Recognition Logger Using Tensorflow
This is an automated facial recognition logger for video security surveillance using TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832).

## Compatibility
The code is tested using TensorFlow 1.12 under Windows 10 with Python 3.6.

## Installation
1. Import bioeye.sql
2. Configure the cameras in config.ini. Example:
```
cam0 = "https://root:ismart12@192.168.1.2/cgi-bin/currentpic.cgi"
log0 = "in_log"
cam1 = "https://root:ismart12@192.168.1.3/cgi-bin/currentpic.cgi"
log1 = "out_log"
```
3. Run bioeye.exe. All cropped faces will be stored in the cluster folder. 
4. Run cluster.py to segregate alike faces.
5. Rename each folder in the train_img folder to identify each person.
6. Press Ctrl+T in bioeye.exe to train the images in the train_img folder.

## Inspiration
The code is heavily inspired by the [Facenet](https://github.com/davidsandberg/facenet) implementation.
