from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from urllib3.exceptions import InsecureRequestWarning
from datetime import datetime, date
from scipy import misc
import mysql.connector
import configparser
import cv2
import facenet
import detect_face
import math
import os
import time
import pickle
import warnings
import contextlib
import requests
import numpy as np
import tensorflow as tf

modeldir = './model/facenet.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
pre_img='./pre_img'
train_img='./train_img'
cluster_dir='./cluster'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="bioeye"
)

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


def main():
    with tf.Graph().as_default():
        last_log=str(datetime.now().time())
        mycursor = mydb.cursor()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            image_size = 182
            input_image_size = 160

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            HumanNames = os.listdir(pre_img)
            HumanNames.sort()

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            # video_capture = cv2.VideoCapture(0)

            print('Start Recognition')
            with no_ssl_verification():
                while True:
                    args = parse_args()

                    # ret, frame = video_capture.read()
                    try:
                        resp = requests.get(args.url)
                        frame = np.asarray(bytearray(resp.content), dtype=np.uint8)
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                        frame_raw = frame
                        if frame_raw.ndim == 2:
                            frame_raw = facenet.to_rgb(frame_raw)
                        img_raw_size = np.asarray(frame_raw.shape)[0:2]
                        frame_raw = frame_raw[:, :, 0:3]
                    except:
                        continue

                    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) #resize frame (optional)

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        rcropped = []
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is very close!')
                                continue

                            rbb = [
                                int((bb[i][0] / img_size[0]) * img_raw_size[0] - (args.margin / 2)),
                                int((bb[i][1] / img_size[1]) * img_raw_size[1] - (args.margin / 2)),
                                int((bb[i][2] / img_size[0]) * img_raw_size[0] + (args.margin / 2)),
                                int((bb[i][3] / img_size[1]) * img_raw_size[1] + (args.margin / 2))
                            ]

                            if rbb[0] < 0:
                                rbb[0] = 0
                            if rbb[1] < 0:
                                rbb[1] = 0
                            if rbb[2] > len(frame_raw[0]):
                                rbb[2] = len(frame_raw[0])
                            if rbb[3] > len(frame_raw):
                                rbb[3] = len(frame_raw)

                            rcropped.append(frame_raw[rbb[1]:rbb[3], rbb[0]:rbb[2], :])
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            try:
                                rcropped[i] = cv2.cvtColor(rcropped[i], cv2.COLOR_RGB2BGR)
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size), interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                print(predictions)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                print(best_class_indices,' with accuracy ',best_class_probabilities)

                                scale_x = 1 / img_size[0]
                                scale_y = 1 / img_size[1]

                                tbb = [ scale_x * bb[i][0], scale_y * bb[i][1], scale_x * bb[i][2], scale_y * bb[i][3] ]

                                bb_w = tbb[2] - tbb[0]
                                bb_h = tbb[3] - tbb[1]
                                bb_area = bb_w * bb_h
                                posY = tbb[1] / 0.5

                                # area = bb_area * 100
                                # area = round(area, 2)
                                # text_x = bb[i][0]
                                # text_y = bb[i][1] - 10
                                # cv2.putText(frame, str(area)+" "+str(round(posY*100,2)), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #             1, (0, 0, 255), thickness=1, lineType=2)

                                if bb_area>args.bb_area:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                                if bb_area>args.bb_area and posY>args.yframe:
                                    if best_class_probabilities>args.class_probability:
                                        #plot result idx under box
                                        text_x = bb[i][0]
                                        text_y = bb[i][3] + 20
                                        print('Result Indices:', best_class_indices[0])
                                        print(HumanNames)

                                        start_time = datetime.strptime(last_log, '%H:%M:%S.%f')
                                        end_time = datetime.strptime(str(datetime.now().time()), '%H:%M:%S.%f')
                                        diff = end_time - start_time
                                        elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))

                                        for H_i in HumanNames:
                                            if HumanNames[best_class_indices[0]] == H_i:
                                                result_names = HumanNames[best_class_indices[0]]
                                                print('Face recognized:', result_names)
                                                if elapsed_time>5000:
                                                    last_log = str(datetime.now().time())
                                                    currdatetime = time.strftime('%Y-%m-%d %H:%M:%S')
                                                    sql = "INSERT INTO "+args.log+" (id_num, date) VALUES (%s, %s)"
                                                    val = (result_names, currdatetime)
                                                    mycursor.execute(sql, val)
                                                    mydb.commit()
                                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                            1, (0, 0, 255), thickness=1, lineType=2)
                                    timestr = time.strftime('%Y%m%d%H%M%S')
                                    misc.imsave(os.path.join(cluster_dir, timestr + '.png'), rcropped[i])
                            except:
                                pass

                    else:
                        print('Alignment Failure')
                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()


def parse_args():
    """Parse input arguments."""
    import argparse
    import configparser

    config = configparser.ConfigParser()

    if config.read('config.ini') == []:
        config['DEFAULT']['margin'] = '44'
        config['DEFAULT']['yframe'] = '0.30'
        config['DEFAULT']['bb_area'] = '0.0025'
        config['DEFAULT']['class_probability'] = '0.53'
        config['DEFAULT']['cluster_update'] = '10000'
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=int, help="cluster image margin", default=int(config['DEFAULT']['margin']))
    parser.add_argument('--yframe', type=float, help="vertical frame limit of face detection", default=float(config['DEFAULT']['yframe']))
    parser.add_argument('--bb_area', type=float, help="bounding box area limit for face detection", default=float(config['DEFAULT']['bb_area']))
    parser.add_argument('--class_probability', type=float, help="face recognition accuracy", default=float(config['DEFAULT']['class_probability']))
    parser.add_argument('-l', '--log', type=str, help='in_log/out_log for database entry', required=True)
    parser.add_argument('-u', '--url', type=str, help='url for image feed', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main()