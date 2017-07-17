# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:54:46 2016
Python Data layer: read data/label, select minibatch and feed it to the network
@author: tb00083
"""

import caffe
from random import shuffle
from caffe_func_utils import mat2py_imdb, mat2py_mean,biproto2py
from caffe_class_utils import lmdbpy, lmdbs
from augmentation import SimpleAugment
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import os.path

class DataLayer(caffe.Layer):
    
  def setup(self,bottom,top):
    self.top_names = ['data', 'label']
    params = eval(self.param_str)
    # Check the paramameters for validity.
    check_params(params)
    # store input as class variables
    
    self.batch_loader = BatchLoader(params)
    self.batch_size = params['batch_size']
    #1
    self.pool = ThreadPool(processes=1)
    self.thread_results = self.pool.apply_async(self.batch_loader.load_next_batch, ())
    
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
    res = self.thread_results.get() #2
    #res = self.batch_loader.load_next_batch()
    
    top[0].data[...] = res['data']#.astype(np.float32,copy = True)
    top[1].data[...] = res['label']#.astype(np.float32,copy=True)
    #3
    self.thread_results = self.pool.apply_async(self.batch_loader.load_next_batch, ())

  
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
  def __init__(self, params):
    #load data
    self.batch_size = params['batch_size']
    self.outshape = params['shape']
    
    if params['source'][-4:] == '.mat':
      self.img_data, self.img_labels = mat2py_imdb(params['source'])
      self.img_data = self.img_data.transpose(2,0,1) #swap dims to have Num x height x width
    elif os.path.isdir(params['source']):  #assume it is lmdb
      lmdb_ = lmdbpy()
      self.img_data, self.img_labels = lmdb_.read(params['source']) #use read_encoded if the img is encoded
      self.img_data = self.img_data.squeeze()
    else:
      assert 0, 'Invalid format for source {}\n\
      need either lmdb or .mat data'.format(params['source'])
    
    label_ids = list(set(self.img_labels))
    NCATS = len(label_ids)
    if label_ids[0]!=0 or label_ids[-1]!=NCATS - 1:
      print 'Your data labels are not [0:{}]. Converting label ...'.format(NCATS-1)
      self.img_labels = [label_ids.index(label) for label in self.img_labels]
    
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
    self.pool = ThreadPool() #4

  def load_next_image(self,l):
    """
    Load the next image in a batch.
    """
    # Did we finish an epoch?
    l.acquire() #5
    if self._cur == len(self.indexlist):
        self._cur = 0
        shuffle(self.indexlist)
    # Load a datum
    index = self.indexlist[self._cur]  # Get the index
    self._cur += 1
    l.release() #6
    
    img = self.img_data[index]
    label = self.img_labels[index]
    return (self.img_augment.augment(img), label)
  
  
  def load_next_batch(self):
    res = {}
    #7
    lock = Lock()
    threads = [self.pool.apply_async(self.load_next_image,(lock,)) for \
                i in range (self.batch_size)]
    thread_res = [thread.get() for thread in threads]
    res['data'] = np.asarray([datum[0] for datum in thread_res])[:,None,:,:]
    res['label'] = np.asarray([datum[1] for datum in thread_res],dtype=np.float32)
    return res
    
#==============================================================================
#     res['data'] = np.zeros((self.batch_size,1,self.outshape[0],self.outshape[1]),dtype = np.float32)
#     res['label'] = np.zeros(self.batch_size,dtype = np.float32)
#     for itt in range(self.batch_size):
#       img, label = self.load_next_image(1)
#       res['data'][itt,...] = img
#       res['label'][itt] = label
#     return res
#==============================================================================

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

def get_batch(batch_loader):
  return batch_loader.load_next_batch()