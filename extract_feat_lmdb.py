# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:09:53 2016

@author: tb00083
extract triplet cnn features
INPUT: deploy.prototxt, weights and input lmdb
OUTPUT: extracted features (.npz or .mat)
"""

CAFFE_DEVICE = 0  #GPU device to be used
layer=0           #last layer in prototxt

domain = 'sketch' # 'image' or 'sketch'
DB = 'Flickr25K/Flickr15K_lmdb/Flickr15k_images'
MEAN_DB = 'models/image_mean.binaryproto'
WEIGHTS = 'models/triplet1_250c80s100i_iter_1000000.caffemodel'

OUT = 'feat_' + domain + '.npz'

"""import libraries"""
import sys
sys.path.insert(1,'./Utils')
import caffe
from caffe_func_utils import extract_cnn_feat

if __name__ == '__main__':
  caffe.set_device(CAFFE_DEVICE)
  caffe.set_mode_gpu()
  
  if domain == 'image':
    DEPLOY = 'models/deploy_images.prototxt'
    scale_factor = 1.0
  else:
    DEPLOY = 'models/deploy_sketch.prototxt'
    scale_factor = 2.0
  
  net_params = dict(DEPLOY_PRO = DEPLOY,
                    WEIGHTS = WEIGHTS,
                    data_mean = MEAN_DB,
                    batch_size = 100,
                    scale_factor = scale_factor)
  
  extract_cnn_feat(net_params, DB, OUT=OUT,layer=layer)
  
  print 'Done.'