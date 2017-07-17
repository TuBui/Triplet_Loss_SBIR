# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:54:46 2016
Data layer for triplet loss
@author: tb00083
"""

import caffe
from random import shuffle
from caffe_func_utils import biproto2py
from caffe_class_utils import lmdbs, svgs
from augmentation import SimpleAugment
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import sys

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
    #1
    self.pool = ThreadPool(processes=1)
    self.thread_results = self.pool.apply_async(\
                            self.batch_loader.load_next_batch, ())
    # reshape
    top[0].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    top[1].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    top[2].reshape(params['batch_size'], 1, params['shape'][0], params['shape'][1])
    #top[3].reshape(params['batch_size'], 3)   #label of anchor,pos & neg example 
    if 'verbose' not in params:
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
    res = self.thread_results.get() #2
    #res = self.batch_loader.load_next_batch()
    
    top[0].data[...] = res['data_a']#.astype(np.float32,copy = True)
    top[1].data[...] = res['data_p']#.astype(np.float32,copy = True)
    top[2].data[...] = res['data_n']#.astype(np.float32,copy = True)
    #top[3].data[...] = res['label'].astype(np.float32,copy=True)
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
    
    self.batch_size = params['batch_size']
    self.outshape = params['shape']
    self.classes_per_batch = params['classes_per_batch']
    
    self.img_lmdb = lmdbs(params['img_source'])
    if params['skt_source'].endswith('.pkl'):
      self.skt_lmdb = svgs(params['skt_source'])
    else:
      self.skt_lmdb = lmdbs(params['skt_source'])
    self.img_labels = self.img_lmdb.get_label_list()
    self.skt_labels = self.skt_lmdb.get_label_list()
    self.img_mean = biproto2py(params['mean_file']).squeeze()
    
    self.num_classes = len(set(self.skt_labels))
    assert len(self.skt_labels)%self.num_classes==0, \
      'Unequal sketch training samples for each class'
    self.skt_per_class = len(self.skt_labels)/self.num_classes
    
    if 'hard_pos' in params:
      self.hard_sel = 1
      self.hard_pos = np.load(params['hard_pos'])['pos']
    elif 'hard_neg' in params:
      self.hard_sel = 2
      self.hard_neg = np.load(params['hard_neg'])['neg']
    elif 'hard_pn' in params:
      self.hard_sel = 3
      tmp = np.load(params['hard_pn'])
      self.hard_pos = tmp['pos']
      self.hard_neg = tmp['neg']
    else: #hard selection turn off
      self.hard_sel = 0
    
    self.img_labels_dict, self.classes = vec2dic(self.img_labels)
    
    self.indexlist = range(len(self.skt_labels))
    self.shuffle_keeping_min_classes_per_batch()
    self._cur = 0  # current image
    
    # this class does some simple data-manipulations
    self.img_augment = SimpleAugment(mean=self.img_mean,shape=params['shape'],
                                     scale = params['scale'], rot = params['rot'])

    if 'verbose' not in params:
      print "BatchLoader initialized with {} sketches, {} images of {} classes".format(
          len(self.skt_labels),
          len(self.img_labels),
          self.num_classes)
    #create threadpools for parallel augmentation
    self.pool = ThreadPool() #4

  def load_next_triplet(self,l):
    """
    Load the next triplet in a batch.
    """
    # Did we finish an epoch?
    l.acquire() #5
    if self._cur == len(self.indexlist):
        self._cur = 0
        self.shuffle_keeping_min_classes_per_batch()    
    # Load a sketch
    index = self.indexlist[self._cur]  # Get the sketch index
    self._cur += 1
    l.release() #6
    
    skt = self.skt_lmdb.get_datum(index).squeeze()
    label = self.skt_labels[index]
    
    #randomly select pos and neg img
    if self.hard_sel == 0:    #hard selection turned off
      index_p = np.random.choice(self.img_labels_dict[str(label)])
      label_n = label
      while label_n == label:
        label_n = np.random.choice(self.classes)
      index_n = np.random.choice(self.img_labels_dict[str(label_n)])
    elif self.hard_sel == 1:  #hard positive selection
      index_p = np.random.choice(self.hard_pos[index])
      label_n = label
      while label_n == label:
        label_n = np.random.choice(self.classes)
      index_n = np.random.choice(self.img_labels_dict[str(label_n)])
    elif self.hard_sel == 2:  #hard neg
      index_p = np.random.choice(self.img_labels_dict[str(label)])
      index_n = np.random.choice(self.hard_neg[index])
    else:   #hard pos and neg
      index_p = np.random.choice(self.hard_pos[index])
      index_n = np.random.choice(self.hard_neg[index])
    
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
    #7
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
#==============================================================================

  def shuffle_keeping_min_classes_per_batch(self):
    shuffle(self.indexlist)

    # sort index list by class
    # (using lambda to restrict sorting to just first element of tuple
    sort_indexlist = sorted(self.indexlist,
          key=lambda index: self.skt_labels[index])
    
    # make it 2D based on classes
    sort_indexlist = np.array(sort_indexlist,dtype=np.int64)
    sort_indexlist = sort_indexlist.reshape(self.num_classes, self.skt_per_class)
    
    # permute the class positions and flat it (make it 1D again)
    sort_indexlist = np.random.permutation(sort_indexlist)
    sort_indexlist = sort_indexlist.reshape(self.num_classes*self.skt_per_class)

    temp_indexlist = np.array([],dtype=np.int64)
    skt_per_batch_per_class = int(self.batch_size/self.classes_per_batch)

    # apply dark magic
    # (slices of classes sliced, appended together until list is over)
    for k in range(0, self.skt_per_class, skt_per_batch_per_class):
      temp_indexlist = np.append(temp_indexlist,
          [
            sort_indexlist[(i*self.skt_per_class) + j]
            for i in range(self.num_classes)
            for j in range(k, k+skt_per_batch_per_class)
          ]
      )

    # convert back to list
    self.indexlist = temp_indexlist.tolist()

def check_params(params):
  """
  A utility function to check the parameters for the data layers.
  """
  required = ['batch_size', 'img_source', 'skt_source','shape','rot',
              'mean_file','scale', 'classes_per_batch']
  for r in required:
      assert r in params.keys(), 'Params must include {}'.format(r)
        

def print_info(name, params):
  """
  Ouput some info regarding the class
  """
  print "{} initialized with settings:\n \
    image source: {}\nsketch source: {}\nmean_file: {}\n\
    batch size: {}, classes per batch: {}, im_shape: {}\n\
    scale: {}\n\
    rotation: {}\n.".format(
      name,
      params['img_source'],
      params['skt_source'],
      params['mean_file'],
      params['batch_size'],
      params['classes_per_batch'],
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