# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:23:28 2016

@author: tb00083
"""

import caffe
from random import shuffle
from caffe_func_utils import mat2py_imdb, mat2py_mean,biproto2py
from augmentation import SimpleAugment
from caffe_class_utils import lmdbpy
import os.path
import numpy as np

class DataLayer(caffe.Layer):
    
  def setup(self,bottom,top):
    self.top_names = ['data', 'label']
    params = eval(self.param_str)
    # Check the paramameters for validity.
    check_params(params)
    # store input as class variables
    self.batch_loader = BatchLoader(params, None)
    self.batch_size = params['batch_size']
    # reshape
    top[0].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    top[1].reshape(params['batch_size'])
    print_info('Data layer',params)
  def reshape(self, bottom, top):
    """
    There is no need to reshape the data, since the input is of fixed size
    (rows and columns)
    """
    pass
  
  def forward(self,bottom,top):
    """
    Load data.
    """
    for itt in range(self.batch_size):
      # Use the batch loader to load the next image.
      img, label = self.batch_loader.load_next_image()

      # Add directly to the caffe data layer
      top[0].data[itt, ...] = img
      top[1].data[itt] = label
  
  def backward(self,bottom,top):
    """
    These layers does not back propagate
    """
    pass


class BatchLoader(object):

  """
  This class abstracts away the loading of images.
  Images can either be loaded singly, or in a batch. The latter is used for
  the asyncronous data layer to preload batches while other processing is
  performed.
  """
  def __init__(self, params, result):
    self.result = result
    
    #load data
    if params['source'][-4:] == '.mat':
      self.img_data, self.img_labels = mat2py_imdb(params['source'])
      self.img_data = np.rollaxis(self.img_data,2)  #swap dims to have Num x height x width
    elif os.path.isdir(params['source']):  #assume it is lmdb
      lmdb_ = lmdbpy()
      self.img_data, self.img_labels = lmdb_.read(params['source']) #use read_encoded if the img is encoded
      self.img_data = self.img_data.squeeze()
    else:
      assert 0, 'Invalid format for source {}\n\
      need either lmdb or .mat data'.format(params['source'])
    
    if params['mean_file']==0:
      self.img_mean = 0
    elif params['mean_file'][-4:] == '.mat':
      self.img_mean = mat2py_mean(params['mean_file'])
    elif params['mean_file'][-12:] == '.binaryproto':
      self.img_mean = biproto2py(params['mean_file']).squeeze()
    else:
      assert 0, 'Invalid format for mean_file {}'.format(params['mean_file'])
    
    
    self.indexlist = range(len(self.img_labels))
    shuffle(self.indexlist)
    self._cur = 0  # current image
    
    # this class does some simple data-manipulations
    self.img_augment = SimpleAugment(mean=self.img_mean,shape=params['shape'],
                                     scale = params['scale'], rot = params['rot'])

    print "\nBatchLoader initialized with {} images".format(
        len(self.img_labels))

  def load_next_image(self):
    """
    Load the next image in a batch.
    """
    # Did we finish an epoch?
    if self._cur == len(self.indexlist):
        self._cur = 0
        shuffle(self.indexlist) 

    # Load a datum
    index = self.indexlist[self._cur]  # Get the index
    img = self.img_data[index]
    label = self.img_labels[index]

    self._cur += 1
    return (self.img_augment.augment(img), label)


def check_params(params):
  """
  A utility function to check the parameters for the data layers.
  """
  required = ['batch_size', 'source','shape','scale','mean_file', 'rot']
  for r in required:
      assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
  """
  Ouput some info regarding the class
  """
  print "\n{} initialized with settings:\nsource: {}\n mean_file: {}\n \
    batch size: {}, im_shape: {}, scale: {}, rotation: {}\n.".format(
      name,
      params['source'],
      params['mean_file'],
      params['batch_size'],
      params['shape'],
      params['scale'],
      params['rot'])