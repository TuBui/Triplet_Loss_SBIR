# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:42:02 2016
model_def hard mining test
@author: tb00083
"""

import caffe
from caffe import layers as L, params as P

from google.protobuf import text_format
import caffe.draw
from caffe.proto import caffe_pb2

##################### define helper functions here #############################
def conv_relu(bottom, kernel_size, num_output, stride=1, pad=0, group=1,
              w_filler = dict(type='gaussian',std = 0.01),
              b_filler = dict(type = 'constant',value = 0.0),
              inplace=True):
  """
  combine convolution & relu in a normal network
  """
  conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                              num_output=num_output, pad=pad, 
                              param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                              weight_filler=w_filler,
                              bias_filler = b_filler)
  return conv, L.ReLU(conv, in_place=inplace)

def fc_relu(bottom, num_output, w_filler = dict(type='gaussian',std = 0.01),
            b_filler = dict(type = 'constant',value = 0.0),
            inplace=True):
  """combine InnerProduct & relu in a normal network"""
  fc = L.InnerProduct(bottom, num_output=num_output, 
                      param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      weight_filler=w_filler,
                      bias_filler = b_filler)
  return fc, L.ReLU(fc, in_place=inplace)

def pooling(bottom, kernel_size, stride = 2, pool = P.Pooling.MAX):
  """quick prototype for pooling layer"""
  return L.Pooling(bottom, kernel_size = kernel_size, stride = stride,
                   pool=pool)

def fullconnect(bottom, num_output, w_filler = dict(type='gaussian',std = 0.01),
      b_filler = dict(type = 'constant',value = 0.0)):
  """quick prototype of fully connected layer"""
  return L.InnerProduct(bottom, num_output = num_output,
                       param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                       bias_filler = b_filler,
                       weight_filler = w_filler)

def conv_relu_triplet(bottom, kernel_size, num_output, stride=1, pad=0, 
                      prefix = 'conv1_a',t=1.0):
  """ for SketchTriplet quick prototyping"""
  conv = L.Convolution(bottom, kernel_size = kernel_size, num_output = num_output, 
                       pad = pad, stride = stride,
                       weight_filler = dict(type='gaussian',std=0.01),
                       bias_filler = dict(type = 'constant',value = 0.0),
                       param = [dict(name=prefix+'_w',lr_mult=1*t,decay_mult=1), 
                                dict(name=prefix+'_b',lr_mult=2*t,decay_mult=0)])
  return conv, L.ReLU(conv, in_place=True)

def conv_relu_triplet_dep(bottom, kernel_size, num_output, stride=1, pad=0):
  """ for SketchTriplet quick prototyping deploy network"""
  conv = L.Convolution(bottom, kernel_size = kernel_size, num_output = num_output, 
                       pad = pad, stride = stride,)
  return conv, L.ReLU(conv, in_place=True)

def fc_relu_triplet(bottom, num_output, prefix = 'fc6_a',t=1.0):
  """ for SketchTriplet quick prototyping"""
  fc = L.InnerProduct(bottom, num_output=num_output, 
                      weight_filler = dict(type='gaussian',std=0.01),
                      bias_filler = dict(type = 'constant',value = 0.0),
                      param = [dict(name=prefix+'_w',lr_mult=1*t,decay_mult=1), 
                                dict(name=prefix+'_b',lr_mult=2*t,decay_mult=0)])
  return fc, L.ReLU(fc, in_place=True)

def fc_relu_triplet_dep(bottom, num_output):
  """ for SketchTriplet quick prototyping deploy network"""
  fc = L.InnerProduct(bottom, num_output=num_output)
  return fc, L.ReLU(fc, in_place=True)

def fc_norm_triplet(bottom, num_output, prefix = 'fc8_a',t=1.0):
  """ for SketchTriplet quick prototyping"""
  fc = L.InnerProduct(bottom, num_output=num_output, 
                      weight_filler = dict(type='gaussian',std=0.01),
                      bias_filler = dict(type = 'constant',value = 0.0),
                      param = [dict(name=prefix+'_w',lr_mult=1*t,decay_mult=1), 
                                dict(name=prefix+'_b',lr_mult=2*t,decay_mult=0)])
  return fc, L.Normalize(fc, in_place=True)

def fc_norm_triplet_dep(bottom, num_output):
  """ for SketchTriplet quick prototyping deploy network"""
  fc = L.InnerProduct(bottom, num_output=num_output)
  return fc, L.Normalize(fc, in_place=True)

def fc_triplet(bottom, num_output, prefix = 'fc8_a',t=1.0):
  """ for SketchTriplet quick prototyping"""
  fc = L.InnerProduct(bottom, num_output=num_output, 
                      weight_filler = dict(type='gaussian',std=0.01),
                      bias_filler = dict(type = 'constant',value = 0.0),
                      param = [dict(name=prefix+'_w',lr_mult=1*t,decay_mult=1), 
                                dict(name=prefix+'_b',lr_mult=2*t,decay_mult=0)])
  return fc

def fc_triplet_dep(bottom, num_output):
  """ for SketchTriplet quick prototyping"""
  fc = L.InnerProduct(bottom, num_output=num_output)
  return fc

##################### define cnn models here ##################################
def LeNet(lmdb, batch_size):
  # our version of LeNet: a series of linear and simple nonlinear transformations
  n = caffe.NetSpec()
  
  n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                           transform_param=dict(scale=1./255), ntop=2)
  
  n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
  n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
  n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
  n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
  n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
  n.relu1 = L.ReLU(n.fc1, in_place=True)
  n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
  n.loss =  L.SoftmaxWithLoss(n.score, n.label)
  
  proto = n.to_proto()
  proto.name = 'LeNet'
  return proto

def SketchANet(data_params, num_class = 20, val_mode = 0):
  """ our version of Sketch-A-Net
  data_params: batch_size, source, shape, scale, rot
  val_mode: 0 if this is train net, 1 if test net, 2 if deploy net
  """
  n = caffe.NetSpec()
  if val_mode == 2:
    n.data        = L.Input(name='data',
                           shape=dict(dim=[1,1,225,225]))
  else:
    n.data, n.label = L.Python(module = 'data_layer', layer = 'DataLayer',
                               ntop = 2, phase = val_mode,
                               param_str = str(data_params))
#==============================================================================
#     n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
#                              transform_param=dict(scale=1./255), ntop=2)
#==============================================================================
  
  n.conv1, n.relu1 = conv_relu(n.data, 15, 64, stride = 3)
  n.pool1 = pooling(n.relu1,3, stride = 2)
  
  n.conv2, n.relu2 = conv_relu(n.pool1, 5, 128)
  n.pool2 = pooling(n.relu2,3, stride = 2)
  
  n.conv3, n.relu3 = conv_relu(n.pool2, 3, 256, pad = 1)
  n.conv4, n.relu4 = conv_relu(n.relu3, 3, 256, 1, 1)
  
  n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1)
  n.pool5 = pooling(n.relu5,3, stride = 2)
  
  n.fc6, n.relu6 = fc_relu(n.pool5, 512)
  if val_mode != 2:
    n.drop6 = L.Dropout(n.relu6, dropout_ratio = 0.55, in_place = True)
    
    n.fc7, n.relu7 = fc_relu(n.drop6, 512)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio = 0.55, in_place = True)
    
    n.fc8 = fullconnect(n.drop7, num_class)
    n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
  else: #deploy mode
    n.fc7, n.relu7 = fc_relu(n.relu6, 512)
    n.fc8 = fullconnect(n.relu7, num_class)
  
  if val_mode==1:
    n.accuracy = L.Accuracy(n.fc8, n.label, phase = val_mode)
  
  proto = n.to_proto()
  proto.name = 'SketchANet'
  return proto

def SketchTriplet_anchor(out_dim):
  n = caffe.NetSpec()
  n.data_a              = L.Input(name='data',
                                  shape=dict(dim=[1,1,225,225]))
  n.conv1_a, n.relu1_a  = conv_relu_triplet_dep(n.data_a, 15, 64, stride = 3)
  n.pool1_a = pooling(n.relu1_a, 3, stride=2)
  
  n.conv2_a, n.relu2_a  = conv_relu_triplet_dep(n.pool1_a, 5, 128)
  n.pool2_a = pooling(n.relu2_a, 3, stride=2)
  
  n.conv3_a, n.relu3_a  = conv_relu_triplet_dep(n.pool2_a, 3, 256)
  
  n.conv4_a, n.relu4_a  = conv_relu_triplet_dep(n.relu3_a, 3, 256)
  
  n.conv5_a, n.relu5_a  = conv_relu_triplet_dep(n.relu4_a, 3, 256)
  n.pool5_a = pooling(n.relu5_a, 3, stride=2)
  
  n.fc6_a, n.relu6_a    = fc_relu_triplet_dep(n.pool5_a, 512)
  
  n.fc7_a, n.relu7_a    = fc_relu_triplet_dep(n.relu6_a, 512)
  
  #n.fc8_a, n.feat_a     = fc_norm_triplet_dep(n.relu7_a, out_dim)
  n.feat_a     = fc_triplet_dep(n.relu7_a, out_dim)
  proto = n.to_proto()
  proto.name = 'SketchTriplet'
  return proto

def SketchTriplet_pos(out_dim):
  n = caffe.NetSpec()
  n.data_p              = L.Input(name='data',
                                  shape=dict(dim=[1,1,225,225]))
  n.conv1_p, n.relu1_p  = conv_relu_triplet_dep(n.data_p, 15, 64, stride = 3)
  n.pool1_p = pooling(n.relu1_p, 3, stride=2)
  
  n.conv2_p, n.relu2_p  = conv_relu_triplet_dep(n.pool1_p, 5, 128)
  n.pool2_p = pooling(n.relu2_p, 3, stride=2)
  
  n.conv3_p, n.relu3_p  = conv_relu_triplet_dep(n.pool2_p, 3, 256)
  
  n.conv4_p, n.relu4_p  = conv_relu_triplet_dep(n.relu3_p, 3, 256)
  
  n.conv5_p, n.relu5_p  = conv_relu_triplet_dep(n.relu4_p, 3, 256)
  n.pool5_p = pooling(n.relu5_p, 3, stride=2)
  
  n.fc6_p, n.relu6_p    = fc_relu_triplet_dep(n.pool5_p, 512)
  
  n.fc7_p, n.relu7_p    = fc_relu_triplet_dep(n.relu6_p, 512)
  
  #n.fc8_p, n.feat_p     = fc_norm_triplet_dep(n.relu7_p, out_dim)
  n.feat_p     = fc_triplet_dep(n.relu7_p, out_dim)
  proto = n.to_proto()
  proto.name = 'SketchTriplet'
  return proto

def SketchTriplet(data_params, loss_params, out_dim = 20, phase = 'train',\
                  share_weights='half',loss_type='old'):
  """SketchTriplet: our triplet network version
  data_params: for Triplet_data layer; incl batch_size, img_source, skt_source
                shape, scale
  loss_params: for triplet_loss layer; incl margin
  phase      : curretnly support 'train' and 'deploy'
  share_weights: must be of type 'full','half', or 'no'
  """
  #wheather anchor shares weights with pos and neg network
  cn = '_p'   #common name added to prefix, applied to pos and neg branch
  if share_weights == 'full':
    an_t = an_b = cn   #anchor name for top and bottom layers
  elif share_weights == 'half':
    an_t = '_a'
    an_b = cn
  else: #no sharing weights
    an_t = an_b = '_a'
  
  #phase: train(0), test(1)
  if phase == 'train' or phase == 'test':
    if phase == 'train':
      mode = 0
    else:
      mode = 1
    
    n = caffe.NetSpec()
    if 'hard_pos' in data_params or 'hard_neg' in data_params or 'hard_pn' in data_params:
      n.data_a, n.data_p, n.data_n = L.Python(name = 'input_data', phase = mode,
                                  ntop=3, module='triplet_data_hardsel',layer = 'TripletDataLayer',
                                  param_str = str(data_params))
    else:
      n.data_a, n.data_p, n.data_n = L.Python(name = 'input_data', phase = mode,
                                  ntop=3, module='triplet_data',layer = 'TripletDataLayer',
                                  param_str = str(data_params))
    #build anchor net
    n.conv1_a, n.relu1_a  = conv_relu_triplet(n.data_a, 15, 64, stride = 3,
                                              prefix = 'conv1'+an_t)
    n.pool1_a = pooling(n.relu1_a, 3, stride=2)
    
    n.conv2_a, n.relu2_a  = conv_relu_triplet(n.pool1_a, 5, 128,
                                              prefix = 'conv2'+an_t)
    n.pool2_a = pooling(n.relu2_a, 3, stride=2)
    
    n.conv3_a, n.relu3_a  = conv_relu_triplet(n.pool2_a, 3, 256, 
                                              prefix = 'conv3'+an_t)
    
    n.conv4_a, n.relu4_a  = conv_relu_triplet(n.relu3_a, 3, 256, 
                                              prefix = 'conv4'+an_b)
    
    n.conv5_a, n.relu5_a  = conv_relu_triplet(n.relu4_a, 3, 256, 
                                              prefix = 'conv5'+an_b)
    n.pool5_a = pooling(n.relu5_a, 3, stride=2)
    
    n.fc6_a, n.relu6_a    = fc_relu_triplet(n.pool5_a, 512, 'fc6'+an_b)
    n.drop6_a = L.Dropout(n.relu6_a, dropout_ratio = 0.55, in_place = True)
    
    n.fc7_a, n.relu7_a    = fc_relu_triplet(n.drop6_a, 512, 'fc7'+an_b)
    n.drop7_a = L.Dropout(n.relu7_a, dropout_ratio = 0.55, in_place = True)
    
    #n.fc8_a, n.feat_a     = fc_norm_triplet(n.drop7_a, out_dim, 'fc8'+an_b)
    n.feat_a     = fc_triplet(n.drop7_a, out_dim, 'fc8'+an_b)
    
    
    #build positive net
    n.conv1_p, n.relu1_p  = conv_relu_triplet(n.data_p, 15, 64, stride = 3,
                                              prefix = 'conv1'+cn)
    n.pool1_p = L.Pooling(n.relu1_p, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.conv2_p, n.relu2_p  = conv_relu_triplet(n.pool1_p, 5, 128, prefix = 'conv2'+cn)
    n.pool2_p = L.Pooling(n.relu2_p, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.conv3_p, n.relu3_p  = conv_relu_triplet(n.pool2_p, 3, 256, prefix = 'conv3'+cn)
    
    n.conv4_p, n.relu4_p  = conv_relu_triplet(n.relu3_p, 3, 256, prefix = 'conv4'+cn)
    
    n.conv5_p, n.relu5_p  = conv_relu_triplet(n.relu4_p, 3, 256, prefix = 'conv5'+cn)
    n.pool5_p = L.Pooling(n.relu5_p, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.fc6_p, n.relu6_p    = fc_relu_triplet(n.pool5_p, 512, 'fc6'+cn)
    n.drop6_p = L.Dropout(n.relu6_p, dropout_ratio = 0.55, in_place = True)
    
    n.fc7_p, n.relu7_p    = fc_relu_triplet(n.drop6_p, 512, 'fc7'+cn)
    n.drop7_p = L.Dropout(n.relu7_p, dropout_ratio = 0.55, in_place = True)
    
    #n.fc8_p, n.feat_p     = fc_norm_triplet(n.drop7_p, out_dim, 'fc8'+cn)
    n.feat_p     = fc_triplet(n.drop7_p, out_dim, 'fc8'+cn)
    
    
    #build negative net
    n.conv1_n, n.relu1_n  = conv_relu_triplet(n.data_n, 15, 64, stride = 3,
                                              prefix = 'conv1'+cn)
    n.pool1_n = L.Pooling(n.relu1_n, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.conv2_n, n.relu2_n  = conv_relu_triplet(n.pool1_n, 5, 128, prefix = 'conv2'+cn)
    n.pool2_n = L.Pooling(n.relu2_n, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.conv3_n, n.relu3_n  = conv_relu_triplet(n.pool2_n, 3, 256, prefix = 'conv3'+cn)
    
    n.conv4_n, n.relu4_n  = conv_relu_triplet(n.relu3_n, 3, 256, prefix = 'conv4'+cn)
    
    n.conv5_n, n.relu5_n  = conv_relu_triplet(n.relu4_n, 3, 256, prefix = 'conv5'+cn)
    n.pool5_n = L.Pooling(n.relu5_n, kernel_size=3, stride=2, pool = P.Pooling.MAX)
    
    n.fc6_n, n.relu6_n    = fc_relu_triplet(n.pool5_n, 512, 'fc6'+cn)
    n.drop6_n = L.Dropout(n.relu6_n, dropout_ratio = 0.55, in_place = True)
    
    n.fc7_n, n.relu7_n    = fc_relu_triplet(n.drop6_n, 512, 'fc7'+cn)
    n.drop7_n = L.Dropout(n.relu7_n, dropout_ratio = 0.55, in_place = True)
    
    #n.fc8_n, n.feat_n     = fc_norm_triplet(n.drop7_n, out_dim, 'fc8'+cn)
    n.feat_n     = fc_triplet(n.drop7_n, out_dim, 'fc8'+cn)
    
    
    #triplet loss
    layer = 'TripletLossLayer' + str(loss_type)
    
    n.loss    = L.Python(n.feat_a, n.feat_p, n.feat_n, 
                         name = 'triplet_loss', loss_weight = loss_params['loss_weight'],
                         module = 'triplet_loss', layer = layer, 
                         param_str = str(loss_params))
    proto = n.to_proto()
    proto.name = 'SketchTriplet'
    return proto
  else:   #deploy
    if share_weights=='full':
      return SketchTriplet_anchor(out_dim)
    else:
      return SketchTriplet_anchor(out_dim),SketchTriplet_pos(out_dim)


def draw_net(net_proto_file, out_img_file, style = 'TB'):
  """
  draw cnn network into image.
  IN:  net_proto_file   net definition file
  IN:  style            'TB' for top-> bottom, 'LR' for lelf->right
  OUT: out_img_file     output image
  """
  net = caffe_pb2.NetParameter()
  text_format.Merge(open(net_proto_file).read(), net)
  if not net.name:
    net.name = 'cnn_net'
  print('\nDrawing net to %s' % out_img_file)
  caffe.draw.draw_net_to_file(net, out_img_file, style)

def make_net():
  data_params = dict(batch_size = 100, source = '/path/to/mat-data',
                     shape = [225,225], scale = 1.0)
  with open('train.prototxt', 'w') as f:
    f.write(str(SketchANet(data_params)))
    
if __name__ == '__main__':
    make_net()