#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:20:57 2017
batch loader simple version.
load batch in order.
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

class batchloader(object):

  """
  load lmdb in mini-batch. Roll-over at the end of the epoch.
  Requirement: NIMGS dividible to batch_size
  Useful in datalayer of CNN training network.
  """
  def __init__(self, params):
    
    self.batch_size = params['batch_size']
    self.outshape = params['shape']
    
    self.lmdb = lmdbs(params['source'])
    self.labels = self.lmdb.get_label_list()
    self.img_mean = biproto2py(params['mean_file']).squeeze()
    
    self.NIMGS = len(self.labels)
    assert self.NIMGS%self.batch_size==0,'NIMGS {} not dividible by batchsize {}'.format(
           self.NIMGS,self.batch_size)
    
    self.num_batches = self.NIMGS/self.batch_size
    self._cur = 0  # current batch
    self.labels_tab = self.labels.reshape((self.num_batches,self.batch_size))
    
    # this class does some simple data-manipulations
    self.img_augment = SimpleAugment(mean=self.img_mean,shape=params['shape'],
                                     scale = params['scale'])
    #create threadpools for parallel augmentation
    #self.pool = ThreadPool() #4
  
  def get_info(self):
    res = dict(NIMGS = self.NIMGS,batch_size = self.batch_size,
               num_batches = self.num_batches,cur = self._cur)
    return res
  def get_labels(self):
    return self.labels
  def IsEpochEnded(self):
    return self._cur == self.num_batches
  
  def load_next_batch(self):
    if self._cur == self.num_batches:
      self._cur = 0
    
    batch_lst = np.arange(self.batch_size) + self._cur * self.batch_size
    chunk = self.img_augment.augment_deploy(self.lmdb.get_data(batch_lst))
    labels = self.labels_tab[self._cur]
    self._cur +=1
  
    return chunk,labels

class batchloader2(object):

  """
  batchloader in general.
  load lmdb in mini-batch. Roll-over at the end of the epoch.
  DIFFERENT: does not require NIMGS dividible to batch_size; hence the last batch
  may have different size.
  Useful in CNN deploy.
  """
  def __init__(self, params):
    
    self.batch_size = params['batch_size']
    self.outshape = params['shape']
    
    self.lmdb = lmdbs(params['source'])
    self.labels = self.lmdb.get_label_list()
    self.img_mean = biproto2py(params['mean_file']).squeeze()
    
    self.NIMGS = len(self.labels)
    
    self.num_batches = int(np.ceil(self.NIMGS/float(self.batch_size)))
    self._cur = 0  # current batch
    
    # this class does some simple data-manipulations
    self.img_augment = SimpleAugment(mean=self.img_mean,shape=params['shape'],
                                     scale = params['scale'])
    #create threadpools for parallel augmentation
    #self.pool = ThreadPool() #4
  
  def get_info(self):
    res = dict(NIMGS = self.NIMGS,batch_size = self.batch_size,
               num_batches = self.num_batches,cur = self._cur)
    return res
  def get_labels(self):
    return self.labels
  def IsEpochEnded(self):
    return self._cur == self.num_batches
  
  def load_next_batch(self):
    if self._cur == self.num_batches:
      self._cur = 0
    batch_lst = np.arange(self._cur * self.batch_size,min([self.NIMGS,(self._cur+1) * self.batch_size]))
    
    chunk = self.img_augment.augment_deploy(self.lmdb.get_data(batch_lst))
    labels = self.labels[batch_lst]
    self._cur +=1
  
    return chunk,labels