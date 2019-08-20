from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
from lxml import etree
import tensorflow as tf
import numpy as np
import os
import sys
import time
import pickle
import argparse
import facenet
import detect_face
from sklearn.cluster import DBSCAN
from datetime import datetime, date

npy='./npy'
pre_img='./pre_img'
model_dir='./model/facenet.pb'
cluster_filename='cluster.xml'
classifier_filename='./class/classifier.pkl'
input_dir='cluster'
output_dir='train_img'
image_size=160
min_cluster_size=1
cluster_threshold=0.8
gpu_memory_fraction=1.0

def main():
    pnet, rnet, onet = create_network_face_detection(gpu_memory_fraction)
    timer = str(datetime.now().time())
    fileCount = 0

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)

            while True:
                args = parse_args()
                time.sleep(0.1)

                newFileCount = len(os.listdir(input_dir))
                start_time = datetime.strptime(timer, '%H:%M:%S.%f')
                end_time = datetime.strptime(str(datetime.now().time()), '%H:%M:%S.%f')
                diff = end_time - start_time
                elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))

                if newFileCount > fileCount and elapsed_time > args.cluster_update:

                    print('Updating cluster...')

                    fileCount = newFileCount
                    timer = str(datetime.now().time())

                    root = etree.Element("root")

                    blacklist = {}
                    try:
                        tree = etree.parse(cluster_filename)
                        for i in tree.iter():
                            if i.tag == 'root':
                                continue
                            if len(i):
                                blacklist.update({i.tag : []})
                                if 'grp-' in i.tag:
                                    for j in i.iter():
                                        if not len(j):
                                            blacklist[i.tag].append([j.tag, j.text])
                    except:
                        pass

                    HumanNames = os.listdir(pre_img)
                    HumanNames.sort()

                    classifier_filename_exp = os.path.expanduser(classifier_filename)
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)

                    image_list = load_images_from_folder(input_dir)
                    images = align_data(image_list, image_size, args.margin, args.bb_area, pnet, rnet, onet)
                    images_placeholder = sess.graph.get_tensor_by_name("input:0")
                    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    embedding_size = embeddings.get_shape()[1]
                    nrof_images = len(images)
                    matrix = np.zeros((nrof_images, nrof_images))

                    tagged = {}
                    for i in range(nrof_images):
                        emb_array = np.zeros((1, embedding_size))
                        emb_array[0, :] = emb[i]
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        if best_class_probabilities>args.class_probability:
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    tagged.update({i: HumanNames[best_class_indices[0]]})

                    # get euclidean distance matrices
                    for i in range(nrof_images):
                        for j in range(nrof_images):
                            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                            matrix[i][j] = dist

                    # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
                    db = DBSCAN(eps=cluster_threshold, min_samples=min_cluster_size, metric='precomputed')
                    db.fit(matrix)
                    labels = db.labels_

                    # get number of clusters
                    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                    print('No of clusters:', no_clusters)

                    if no_clusters > 0:
                        for i in range(no_clusters):
                            for j in np.nonzero(labels == i)[0]:
                                # path = os.path.join(output_dir, (tagged[j] if j in tagged else str(i)))
                                # if not os.path.exists(path):
                                #     os.makedirs(path)
                                # misc.imsave(os.path.join(path, image_list[j][0]), image_list[j][1])
                                tag = ("grp-" + tagged[j] if j in tagged else "unk-" + str(i))
                                group = root.find(tag)

                                exclude = False
                                for x in blacklist:
                                    for y in blacklist[x]:
                                        if y[1] == image_list[j][0]:
                                            exclude = True
                                            break
                                    if exclude:
                                        break

                                if not exclude:
                                    if group is None:
                                        group = etree.SubElement(root, tag)
                                    etree.SubElement(group, "face" + str(j)).text = image_list[j][0]

                    for x in blacklist:
                        for y in blacklist[x]:
                            group = root.find(x)
                            if group is None:
                                group = etree.SubElement(root, x)
                            etree.SubElement(group, y[0]).text = y[1]

                    tree = etree.ElementTree(root)
                    tree.write(cluster_filename)



def align_data(image_list, image_size, margin, area, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []

    for x in range(len(image_list)):
        if image_list[x][1].ndim == 2:
            image_list[x][1] = facenet.to_rgb(image_list[x][1])
        image_list[x][1] = image_list[x][1][:, :, 0:3]
        aligned = misc.imresize(image_list[x][1], (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        prewhitened.reshape(-1,image_size,image_size,3)
        img_list.append(prewhitened)

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append([filename, img])
    return images


def parse_args():
    """Parse input arguments."""
    import argparse
    import configparser
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=int, help="cluster image margin", default=int(config['DEFAULT']['margin']))
    parser.add_argument('--bb_area', type=float, help="bounding box area limit for face detection", default=float(config['DEFAULT']['bb_area']))
    parser.add_argument('--class_probability', type=float, help="face recognition accuracy", default=float(config['DEFAULT']['class_probability']))
    parser.add_argument('--cluster_update', type=int, help="cluster udpate rate (ms)", default=int(config['DEFAULT']['cluster_update']))
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main()