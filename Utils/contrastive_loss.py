# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:52:58 2016
contrastive loss
@author: tb00083
"""

import caffe
import numpy as np
#import ipdb

class ContrastiveLossLayer0(caffe.Layer):
  """
  Contrastive loss with Eucludean distance
  L = 1/2N * [(1-y)*(a-b)^2 + y*max{0,m - ||a-b||}^2 
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 4:
      raise Exception("Need 4 inputs to compute contrastive loss.")
    params = eval(self.param_str)
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) #cached a-b
    self.dist_sq = np.zeros((bottom[0].num,1), dtype=np.float32) #cached ||a-b||^2
    self.dist = np.zeros_like(self.dist_sq, dtype=np.float32) #cache ||a-b||
    self.msub = np.zeros_like(self.dist, dtype=np.float32) #cached max{0,m - ||a-b||}
    self.loss   = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.eps = np.finfo(np.float32).eps
    self.y = np.float32(bottom[2].data != bottom[3].data)

  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count:
      raise Exception("Inputs must have the same dimension.")
    # difference is shape of inputs
    
#==============================================================================
#         shape = bottom[0].shape
#         self.loss = np.zeros((shape[0], shape[2], shape[3]))
#==============================================================================
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = bottom[0].data - bottom[1].data
    self.y = np.float32(bottom[2].data != bottom[3].data)
    #self.dist_sq[...] = np.sum(self.diff*self.diff,axis=1,keepdims=True) #if it complaint dim mismatch, switch to the following line
    self.dist_sq[...] = np.sum(self.diff.squeeze()**2,axis=1,keepdims=True)
    self.dist[...] = np.sqrt(self.dist_sq)
    self.msub[...] = np.maximum(0,self.margin -self.dist)
    
    self.loss[...] = (1 - self.y)*self.dist_sq +\
                self.y * self.msub*self.msub
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = (self.msub>0).astype(np.float32)
    
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = (1-self.y - \
              active*self.y*(self.margin/(self.dist+self.eps)-1))*\
                          self.diff / bottom[0].num *self.loss_weight
#==============================================================================
#       bottom[0].diff[...] = (1-self.y - \
#               active*self.y*(self.margin/(self.dist+self.eps)-1))[:,:,None,None]*\
#                           self.diff / bottom[0].num
#==============================================================================
    if propagate_down[1]:
      bottom[1].diff[...] = -bottom[0].diff
    
#==============================================================================
#     if active.sum()/active.size < 0.5:
#       print '# active triplets is small {}/{}'.format(active.sum(),active.size)
#       self.margin *= 2
#       print 'double margin to {}'.format(self.margin)
#==============================================================================

class ContrastiveLossLayer1(caffe.Layer):
  """
  Contrastive loss with Eucludean distance
  L = 1/2N * [(1-y)*(2a-b)^2 + y*max{0,m - ||2a-b||}^2 
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 3:
      raise Exception("Need 3 inputs to compute contrastive loss.")
    params = eval(self.param_str)
    
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) #cached a-b
    self.dist_sq = np.zeros((bottom[0].num,1), dtype=np.float32) #cached ||a-b||^2
    self.dist = np.zeros_like(self.dist_sq, dtype=np.float32) #cache ||a-b||
    self.msub = np.zeros_like(self.dist, dtype=np.float32) #cached max{0,m - ||a-b||}
    self.loss   = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.eps = np.finfo(np.float32).eps
    self.tmp = np.zeros_like(bottom[0].data, dtype=np.float32) #cached tmp for backward
    self.y = np.float32(bottom[2].data != bottom[3].data)

#==============================================================================
#     self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) #cached a-b
#     self.dist_sq = np.zeros((bottom[0].num,1,1,1), dtype=np.float32) #cached ||a-b||^2
#     self.dist = np.zeros_like(self.dist_sq, dtype=np.float32) #cache ||a-b||
#     self.msub = np.zeros_like(self.dist, dtype=np.float32) #cached max{0,m - ||a-b||}
#     self.loss   = np.zeros((bottom[0].num,1,1,1), dtype=np.float32)
#     self.eps = np.finfo(np.float32).eps
#     self.tmp = np.zeros_like(bottom[0].data, dtype=np.float32) #cached tmp for backward
# 
#==============================================================================
  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count:
      raise Exception("Inputs must have the same dimension.")
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 2*bottom[0].data - bottom[1].data
    self.y = np.float32(bottom[2].data != bottom[3].data)
    self.dist_sq[...] = np.sum(self.diff*self.diff,axis=1,keepdims=True)
    self.dist[...] = np.sqrt(self.dist_sq)
    self.msub[...] = np.maximum(0,self.margin -self.dist)
    
    self.loss[...] = (1 - self.y)*self.dist_sq +\
                self.y * self.msub*self.msub
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = (self.msub>0).astype(np.float32)
    self.tmp[...] = self.diff*(1-self.y-active*self.y*\
                    (self.margin/(self.dist+self.eps)-1.0)) / bottom[0].num
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = 2*self.tmp *self.loss_weight
    if propagate_down[1]:
      bottom[1].diff[...] = -self.tmp *self.loss_weight

class ContrastiveLossLayer2(caffe.Layer):
  """
  Contrastive loss with Eucludean distance
  L = 1/2N * [(1-y)*(2a-b)^2 + y*max{0,m - (2a-b)^2}
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 3:
      raise Exception("Need 3 inputs to compute contrastive loss.")
    params = eval(self.param_str)
    
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) #cached a-b
    self.dist_sq = np.zeros((bottom[0].num,1), dtype=np.float32) #cached ||a-b||^2
    #self.dist = np.zeros_like(self.dist_sq, dtype=np.float32) #cache ||a-b||
    self.msub = np.zeros_like(self.dist_sq, dtype=np.float32) #cached max{0,m - ||a-b||}
    self.loss   = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.eps = np.finfo(np.float32).eps
    self.tmp = np.zeros_like(bottom[0].data, dtype=np.float32) #cached tmp for backward
    self.y = np.float32(bottom[2].data != bottom[3].data)

#==============================================================================
#     self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) #cached a-b
#     self.dist_sq = np.zeros((bottom[0].num,1,1,1), dtype=np.float32) #cached ||a-b||^2
#     self.dist = np.zeros_like(self.dist_sq, dtype=np.float32) #cache ||a-b||
#     self.msub = np.zeros_like(self.dist, dtype=np.float32) #cached max{0,m - ||a-b||}
#     self.loss   = np.zeros((bottom[0].num,1,1,1), dtype=np.float32)
#     self.eps = np.finfo(np.float32).eps
#     self.tmp = np.zeros_like(bottom[0].data, dtype=np.float32) #cached tmp for backward
# 
#==============================================================================
  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count:
      raise Exception("Inputs must have the same dimension.")
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 2*bottom[0].data - bottom[1].data
    self.y = np.float32(bottom[2].data != bottom[3].data)
    self.dist_sq[...] = np.sum(self.diff*self.diff,axis=1,keepdims=True)
    #self.dist[...] = np.sqrt(self.dist_sq)
    self.msub[...] = np.maximum(0,self.margin -self.dist_sq)
    
    self.loss[...] = (1 - self.y)*self.dist_sq +\
                self.y * self.msub
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = (self.msub>0).astype(np.float32)
    self.tmp[...] = self.diff*(1-self.y-active*self.y) / bottom[0].num
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = 2*self.tmp *self.loss_weight
    if propagate_down[1]:
      bottom[1].diff[...] = -self.tmp *self.loss_weight
