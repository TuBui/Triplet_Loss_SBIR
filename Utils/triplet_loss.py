import caffe
import numpy as np
#import ipdb

class TripletLossLayer0(caffe.Layer):
  """
  Compute the triplet Loss in the same manner as the C++ EuclideanLossLayer
  to demonstrate the class interface for developing layers in Python.
  L = 1/2N * max{0, m + |a-p|^2 - |a-n|^2}
  last change: if norm layer is added before this layer; it cause dim change 
    from Nx1 to Nx1x1x1, we need to adapt both cases (norm or no-norm here)
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 3:
      raise Exception("Need 3 inputs to compute triplet loss.")
    params = eval(self.param_str)
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']
    
    self.diff_p = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.diff_p_sq = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.diff_n = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.diff_n_sq = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.loss   = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.exdim = len(bottom[0].data.shape) - len(self.loss.shape) #if norm layer in the network

  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count or bottom[0].count != bottom[2].count:
      raise Exception("Inputs must have the same dimension.")
    
#==============================================================================
#         shape = bottom[0].shape
#         self.loss = np.zeros((shape[0], shape[2], shape[3]))
#==============================================================================
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff_p[...] = bottom[0].data - bottom[1].data
    self.diff_n[...] = bottom[0].data - bottom[2].data
#==============================================================================
#     self.diff_p_sq[...] = np.sum(self.diff_p**2,axis=1,keepdims=True)
#     self.diff_n_sq[...] = np.sum(self.diff_n**2,axis=1,keepdims=True)
#==============================================================================
    self.diff_p_sq[...] = np.sum(self.diff_p.squeeze()**2,axis=1,keepdims=True) #if norm layer...
    self.diff_n_sq[...] = np.sum(self.diff_n.squeeze()**2,axis=1,keepdims=True)
    self.loss[...] = np.maximum(0,self.margin + self.diff_p_sq - self.diff_n_sq)
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = np.zeros_like(self.loss,dtype=np.float32)
    active[np.where(self.loss > 0.0)] = 1.0
    for i in range(self.exdim): #if norm layer in the network
      active = active[...,None]
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = self.loss_weight*(self.diff_p - self.diff_n)*active / bottom[0].num
    if propagate_down[1]:
      bottom[1].diff[...] = -self.loss_weight*self.diff_p*active / bottom[1].num
    if propagate_down[2]:
      bottom[2].diff[...] = self.loss_weight*self.diff_n*active / bottom[2].num
    
#==============================================================================
#     if active.sum()/active.size < 0.5:
#       print '# active triplets is small {}/{}'.format(active.sum(),active.size)
#       self.margin *= 2
#       print 'double margin to {}'.format(self.margin)
#==============================================================================

class TripletLossLayer1(caffe.Layer):
  """
  Compute the triplet Loss in the same manner as the C++ EuclideanLossLayer
  to demonstrate the class interface for developing layers in Python.
  L = 1/2N * max{0, m + |sf*a-p|^2 - |sf*a-n|^2}
  sf is the anchor scale factor. at sf=1.0 it becomes TripletLossLayer0
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 3:
      raise Exception("Need 3 inputs to compute triplet loss.")
    params = eval(self.param_str)
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']
    self.sf = params['anchor_scale']
    print('Loss layer constructed with:\n  margin:{}\n  loss_weight: {}\n  anchor scale: {}\n'.format(
          self.margin,self.loss_weight,self.sf))
    self.diff_p = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.diff_p_sq = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.diff_n = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.diff_n_sq = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.loss   = np.zeros((bottom[0].num,1), dtype=np.float32)
    self.exdim = len(bottom[0].data.shape) - len(self.loss.shape) #if norm layer in the network

  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count or bottom[0].count != bottom[2].count:
      raise Exception("Inputs must have the same dimension.")
    
#==============================================================================
#         shape = bottom[0].shape
#         self.loss = np.zeros((shape[0], shape[2], shape[3]))
#==============================================================================
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff_p[...] = self.sf * bottom[0].data - bottom[1].data
    self.diff_n[...] = self.sf * bottom[0].data - bottom[2].data
    self.diff_p_sq[...] = np.sum(self.diff_p.squeeze()**2,axis=1,keepdims=True) #if norm layer...
    self.diff_n_sq[...] = np.sum(self.diff_n.squeeze()**2,axis=1,keepdims=True)
    self.loss[...] = np.maximum(0,self.margin + self.diff_p_sq - self.diff_n_sq)
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = np.zeros_like(self.loss,dtype=np.float32)
    active[np.where(self.loss > 0.0)] = 1.0
    for i in range(self.exdim): #if norm layer in the network
      active = active[...,None]
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = self.sf*self.loss_weight*(self.diff_p - self.diff_n)*active / bottom[0].num
    if propagate_down[1]:
      bottom[1].diff[...] = -self.loss_weight*self.diff_p*active / bottom[1].num
    if propagate_down[2]:
      bottom[2].diff[...] = self.loss_weight*self.diff_n*active / bottom[2].num

class TripletLossLayer2(caffe.Layer):
  """
  Compute the triplet Loss in the same manner as the C++ EuclideanLossLayer
  to demonstrate the class interface for developing layers in Python.
  Loss function
  L = 1/2 * max{0, 1 - ||a-n||^2/(m+||a-p||^2)}
  """

  def setup(self, bottom, top):
    # check input pair
    if len(bottom) != 3:
      raise Exception("Need 3 inputs to compute triplet loss.")
    params = eval(self.param_str)
    self.margin = params['margin']
    self.loss_weight = params['loss_weight']

  def reshape(self, bottom, top):
    # check input dimensions match
    if bottom[0].count != bottom[1].count or bottom[0].count != bottom[2].count:
      raise Exception("Inputs must have the same dimension.")
    # difference is shape of inputs
    self.diff_p = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.diff_n = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.loss   = np.zeros_like(bottom[0].data, dtype=np.float32)
#==============================================================================
#         shape = bottom[0].shape
#         self.loss = np.zeros((shape[0], shape[2], shape[3]))
#==============================================================================
    # loss output is scalar
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff_p[...] = bottom[0].data - bottom[1].data
    self.diff_n[...] = bottom[0].data - bottom[2].data
    self.diff_sq_p = self.diff_p**2
    self.diff_sq_n = self.diff_n**2
    dp = np.sum(self.diff_sq_p,axis=1,keepdims=True)
    dn = np.sum(self.diff_sq_n,axis=1,keepdims=True)
    self.loss = np.maximum(0,1.0 - dn/(self.margin + dp))
    top[0].data[...] = np.sum(self.loss) / bottom[0].num /2.0
    #ipdb.set_trace()

  def backward(self, top, propagate_down, bottom):
    active = np.zeros_like(self.loss,dtype=np.float32)
    active[np.where(self.loss > 0.0)] = 1.0
    diff_pn = bottom[1].data - bottom[2].data
    #ipdb.set_trace()
    if propagate_down[0]:
      bottom[0].diff[...] = self.loss_weight*self.diff_n*(self.diff_p*diff_pn-self.margin)/((self.margin+self.diff_sq_p)**2) * active / bottom[0].num
    if propagate_down[1]:
      bottom[1].diff[...] = -self.loss_weight*self.diff_sq_n*self.diff_p/((self.margin+self.diff_sq_p)**2) *active / bottom[1].num
    if propagate_down[2]:
      bottom[2].diff[...] = self.loss_weight*self.diff_n/(self.margin+self.diff_sq_p) *active / bottom[2].num