from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import shutil
from lxml import etree
from classifier import training
from preprocess import preprocesses

input_datadir = './train_img'
output_datadir = './pre_img'
clusterdir = './cluster'
modeldir = './model/facenet.pb'
cluster_filename = 'cluster.xml'
classifier_filename = './class/classifier.pkl'

# grab faces tagged with their id number
folders = {}
tree = etree.parse(cluster_filename)
for i in tree.iter():
	if i.tag == 'root':
		continue
	if len(i) and 'grp-' in i.tag:
		key = i.tag.replace('grp-', '')
		folders.update({key : []})
		for j in i.iter():
		    if not len(j):
		        folders[key].append(j.text)

# move the images from cluster to the training folder
for x in folders:
	for y in folders[x]:
		src = clusterdir+'/'+y
		dst = input_datadir+'/'+x
		if os.path.exists(src):
			if not os.path.exists(dst):
				os.mkdir(dst)
			os.rename(src, dst+'/'+y)

# update the cluster
root = tree.getroot()
for f in folders:
	f = 'grp-'+f
	root.remove(root.find(f))
tree.write(cluster_filename)

# preprocess data
obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

# start training
print ("Training Start")
obj=training(output_datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)

# delete training data
print('Deleting training data...')
for the_file in os.listdir(input_datadir):
    file_path = os.path.join(input_datadir, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
        	shutil.rmtree(file_path)
    except Exception as e:
        print(e)

sys.exit("All Done")