# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:36 2017

@author: tb00083
"""

import caffe
import numpy as np

class TripletSelectLayer(caffe.Layer):
  
  def setup(self,bottom,top):
    if len(bottom) != 4:
      raise Exception("Need 4 inputs in TripletSelectLayer.")
    params = eval(self.param_str)
    self.asf = params['asf']
    self.hm_type = params['hm_type']
    
    top[0].reshape(*bottom[0].data.shape)
    top[1].reshape(*bottom[0].data.shape)
    top[2].reshape(*bottom[0].data.shape)
    self.batchsize = bottom[0].data.shape[0]
    #self.triplet_list = np.zeros((self.batchsize,3),dtype = np.uint64)
    
  def reshape(self, bottom, top):
    """
    
    """
    pass
  
  def forward(self,bottom,top):
    """
    
    """
    feat_a = np.array(bottom[0].data).squeeze()
    feat_i = np.array(bottom[1].data).squeeze()
    label_a = np.array(bottom[2].data).squeeze()[...,None] #size Nx1
    label_i = np.array(bottom[3].data).squeeze()
    
    #retrieval within mini batch
    pdist = [np.sum((feat_a[i]*self.asf-feat_i)**2,axis=1) for i in range(self.batchsize)]
    retid = np.array(pdist).argsort()
    ret_labels = label_i[retid]
    rel = ret_labels == label_a
    
    #hard pos and neg
    pos = [retid[i][rel[i]][::-1] for i in range(self.batchsize)]
    neg = [retid[i][~rel[i]]      for i in range(self.batchsize)]
    
    if self.hm_type == 'hard_neg':
      self.triplet_list = [(i, np.random.choice(pos[i]), neg[i][0]) for i in range(self.batchsize)]
    elif self.hm_type == 'hard_pos':
      self.triplet_list = [(i, pos[i][0], np.random.choice(neg[i])) for i in range(self.batchsize)]
    else: #assume both hard pos and neg
      self.triplet_list = [(i, pos[i][0], neg[i][0]) for i in range(self.batchsize)]
    
    #output anchor, pos and neg blobs
    top[0].data[...] = bottom[0].data
    top_pos = np.array([self.triplet_list[i][1] for i in range(self.batchsize)])
    top_neg = np.array([self.triplet_list[i][2] for i in range(self.batchsize)])
    top[1].data[...] = bottom[1].data[top_pos,...]
    top[2].data[...] = bottom[1].data[top_neg,...]
  
  def backward(self, top, propagate_down, bottom):
      """
      
      """
      if propagate_down[0]:
        bottom[0].diff[...] = top[0].diff
      if propagate_down[1]:
        bottom_diff = np.zeros(bottom[1].data.shape)
        for i in xrange(self.batchsize):
            bottom_diff[self.triplet_list[i][1]] += top[1].diff[i]
            bottom_diff[self.triplet_list[i][2]] += top[2].diff[i]
        bottom[1].diff[...] = bottom_diff