# -*- coding=utf-8 -*-

import tensorflow as tf
import os
dic = {}
tf.app.flags.DEFINE_string('data_dir', '/mnt/shared/easydl/train_data_dir', '''原始训练图片路径''')
tf.app.flags.DEFINE_string('output_dir', '/mnt/shared/easydl/train_data_dir', '''原始训练图片路径''')

FLAGS = tf.app.flags.FLAGS

data_dir = FLAGS.data_dir+'images'+'/'
output_dir = FLAGS.output_dir

for i in tf.gfile.ListDirectory(data_dir):
    label = i.rsplit('_',1)

    if label[0] in dic.keys():
        dic[label[0]].append(i)
    else:
        dic[label[0]]=[i]
#print(dic)
for i in dic:
    path = output_dir+i
    print(path)
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)
    for j in dic[i]:
        if not tf.gfile.IsDirectory(j):
           print(tf.gfile.Copy(data_dir+j,path+'/'+j))
'''
for i in tf.gfile.Glob(data_dir+'/'+'*.*'):
    tf.gfile.Remove(i)
'''