# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:54:46 2016
Data layer for triplet loss
@author: tb00083
"""

import caffe
from random import shuffle
from caffe_func_utils import biproto2py
from caffe_class_utils import lmdbs
from augmentation import SimpleAugment
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock

class TripletDataLayer(caffe.Layer):
  
  def setup(self,bottom,top):
    #self.top_names = ['data_a', 'data_p', 'data_n', 'data_l']
    self.top_names = ['data_a', 'data_p', 'data_n']
    params = eval(self.param_str)
    # Check the paramameters for validity.
    check_params(params)
    # store input as class variables
    self.batch_loader = BatchLoader(params)
    self.batch_size = params['batch_size']
    self.pool = ThreadPool(processes=1)
    self.thread_results = self.pool.apply_async(\
                            self.batch_loader.load_next_batch, ())
    # reshape
    top[0].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    top[1].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    top[2].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    #top[3].reshape(params['batch_size'], 3)   #label of anchor,pos & neg example 
    print_info('Triplet data layer',params)
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
    res = self.thread_results.get()
    #res = self.batch_loader.load_next_batch()
    top[0].data[...] = res['data_a'].astype(np.float32,copy = True)
    top[1].data[...] = res['data_p'].astype(np.float32,copy = True)
    top[2].data[...] = res['data_n'].astype(np.float32,copy = True)
    #top[3].data[...] = res['label'].astype(np.float32,copy=True)
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
    
    self.batch_size = params['batch_size']
    self.outshape = params['shape']
    
    
    self.img_lmdb = lmdbs(params['img_source'])
    self.skt_lmdb = lmdbs(params['skt_source'])
    self.img_labels = self.img_lmdb.get_label_list()
    self.skt_labels = self.skt_lmdb.get_label_list()
    self.img_mean = biproto2py(params['mean_file']).squeeze()
    
    self.img_labels_dict, self.classes = vec2dic(self.img_labels)
    self.indexlist = range(len(self.skt_labels))
    shuffle(self.indexlist)
    self._cur = 0  # current image
    
    # this class does some simple data-manipulations
    self.img_augment = SimpleAugment(mean=self.img_mean,shape=params['shape'],
                                     scale = params['scale'], rot = params['rot'])

    print "BatchLoader initialized with {} sketches, {} images".format(
        len(self.skt_labels),
        len(self.img_labels))
    #create threadpools for parallel augmentation
    self.pool = ThreadPool()

  def load_next_triplet(self,l):
    """
    Load the next triplet in a batch.
    """
    # Did we finish an epoch?
    l.acquire()
    if self._cur == len(self.indexlist):
        self._cur = 0
        shuffle(self.indexlist)
    
    # Load a sketch
    index = self.indexlist[self._cur]  # Get the sketch index
    self._cur += 1
    l.release()
    
    skt = self.skt_lmdb.get_datum(index).squeeze()
    label = self.skt_labels[index]
    
    #randomly select pos and neg img
    index_p = np.random.choice(self.img_labels_dict[str(label)])
    label_n = label
    while label_n == label:
      label_n = np.random.choice(self.classes)
    index_n = np.random.choice(self.img_labels_dict[str(label_n)])
    img_p   = self.img_lmdb.get_datum(index_p).squeeze()
    img_n   = self.img_lmdb.get_datum(index_n).squeeze()
    
    res = dict(anchor=self.img_augment.augment(skt)
               ,pos = self.img_augment.augment(img_p)
               ,neg = self.img_augment.augment(img_n)
               ,label = np.array([label, self.img_labels[index_p]
               ,self.img_labels[index_n]]))
    return res

  def load_next_batch(self):
    res = {}
    lock = Lock()
    threads = [self.pool.apply_async(self.load_next_triplet,(lock,)) for \
                i in range (self.batch_size)]
    thread_res = [thread.get() for thread in threads]
    res['data_a'] = np.asarray([tri['anchor'] for tri in thread_res])[:,None,:,:]
    res['data_p'] = np.asarray([tri['pos'] for tri in thread_res])[:,None,:,:]
    res['data_n'] = np.asarray([tri['neg'] for tri in thread_res])[:,None,:,:]
    res['label'] = np.asarray([tri['label'] for tri in thread_res],dtype=np.float32)
    return res
#==============================================================================
#     res['data_a'] = np.zeros((self.batch_size,1,self.outshape[0],\
#                             self.outshape[1]),dtype = np.float32)
#     res['data_p'] = np.zeros_like(res['data_a'],dtype=np.float32)
#     res['data_n'] = np.zeros_like(res['data_a'],dtype=np.float32)
#     res['label'] = np.zeros((self.batch_size,3),dtype = np.float32)
#     for itt in range(self.batch_size):
#       trp = self.load_next_triplet(1)
#       res['data_a'][itt,...] = trp['anchor']
#       res['data_p'][itt,...] = trp['pos']
#       res['data_n'][itt,...] = trp['neg']
#       res['label'][itt,...] = trp['label']
#     return res
# 
#==============================================================================

def check_params(params):
  """
  A utility function to check the parameters for the data layers.
  """
  required = ['batch_size', 'img_source', 'skt_source','shape','rot',
              'mean_file','scale']
  for r in required:
      assert r in params.keys(), 'Params must include {}'.format(r)
        

def print_info(name, params):
  """
  Ouput some info regarding the class
  """
  print "{} initialized with settings:\n \
    image source: {}\nsketch source: {}\nmean_file: {}\n\
    batch size: {}, im_shape: {}, scale: {}, rotation: {}\n.".format(
      name,
      params['img_source'],
      params['skt_source'],
      params['mean_file'],
      params['batch_size'],
      params['shape'],
      params['scale'],
      params['rot'])

def vec2dic(vec):
  """Convert numpy vector to dictionary where elements with same values 
  are grouped together.
  e.g. vec = [1 2 1 4 4 4 3] -> output = {'1':[0,2],'2':1,'3':6,'4':[3,4,5]}
  """
  vals = np.unique(vec)
  dic = {}
  for v in vals:
    dic[str(v)] = [i for i in range(len(vec)) if vec[i] == v]
  return dic, vals
