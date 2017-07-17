# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:42:02 2016
new version of model_def, designed for half-share network with pretrain branches
and standard loss function
@author: tb00083
"""

import caffe
from caffe import layers as L, params as P

from google.protobuf import text_format
import caffe.draw
from caffe.proto import caffe_pb2
import copy
Train_Mode = dict(train = 0, test = 1, deploy = 2)

##################### define helper functions here #############################
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
learned_param2 = [dict(lr_mult=2, decay_mult=1),dict(lr_mult=4, decay_mult=0)]

frozen_param = [dict(lr_mult=0), dict(lr_mult=0)]
                
def conv_relu(bottom, kernel_size, num_output, stride=1, pad=0, group=1, \
              w_filler = dict(type='gaussian',std = 0.01), \
              b_filler = dict(type = 'constant',value = 0.0), \
              param = learned_param, \
              name_prefix = '', \
              inplace=True):
  """
  combine convolution & relu in a normal network
  """
  pr = copy.deepcopy(param)
  if name_prefix:  #specify param name for wweight sharing
    pr[0]['name'] = name_prefix + '_w'
    pr[1]['name'] = name_prefix + '_b'
    
  conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                              num_output=num_output, pad=pad, 
                              param = pr, group = group, 
                              weight_filler=w_filler,
                              bias_filler = b_filler)
  return conv, L.ReLU(conv, in_place=inplace)

def fc_relu(bottom, num_output, w_filler = dict(type='gaussian',std = 0.01), \
            b_filler = dict(type = 'constant',value = 0.0), \
            param = learned_param, \
            name_prefix = '', \
            inplace=True):
  """combine InnerProduct & relu in a normal network"""
  pr = copy.deepcopy(param)
  if name_prefix:  #specify param name for wweight sharing
    pr[0]['name'] = name_prefix + '_w'
    pr[1]['name'] = name_prefix + '_b'
    
  fc = L.InnerProduct(bottom, num_output=num_output, 
                      param = pr,
                      weight_filler=w_filler,
                      bias_filler = b_filler)
  return fc, L.ReLU(fc, in_place=inplace)

def pooling(bottom, kernel_size, stride = 2, pool = P.Pooling.MAX):
  """quick prototype for pooling layer"""
  return L.Pooling(bottom, kernel_size = kernel_size, stride = stride,
                   pool=pool)

def norm(bottom, local_size=5, alpha=1e-4, beta=0.75):
  """quick prototype of LRN normalisation"""
  return L.LRN(bottom, local_size=local_size, \
         alpha=alpha, beta=beta)

def fullconnect(bottom, num_output,\
      w_filler = dict(type='gaussian',std = 0.01),
      b_filler = dict(type = 'constant',value = 0.0),\
      param=learned_param,\
      name_prefix=''):
  """quick prototype of fully connected layer"""
  pr = copy.deepcopy(param)
  if name_prefix:  #specify param name for wweight sharing
    pr[0]['name'] = name_prefix + '_w'
    pr[1]['name'] = name_prefix + '_b'
  return L.InnerProduct(bottom, num_output = num_output,
                       param = pr,
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
  n.norm_a = L.Normalize(n.feat_a,in_place=True)
  
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
  n.norm_p = L.Normalize(n.feat_p,in_place=True)
  
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
    n.norm_a = L.Normalize(n.feat_a,in_place=True)
    
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
    n.norm_p = L.Normalize(n.feat_p,in_place=True)
    
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
    n.norm_n = L.Normalize(n.feat_n,in_place=True)
    
    #triplet loss
    layer = 'TripletLossLayer' + str(loss_type)
    
    n.loss    = L.Python(n.norm_a, n.norm_p, n.norm_n, 
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

########################################half-share with pretrain###############
def pretrain_sketch(data_params, num_class = 20, mode = 'train',learn_all=True):
  """ our version of Sketch-A-Net
  data_params: batch_size, source, shape, scale, rot
  val_mode: 0 if this is train net, 1 if test net, 2 if deploy net
  """
  param = learned_param if learn_all else frozen_param
  n = caffe.NetSpec()
  if mode == 'deploy':
    n.data        = L.Input(name='data',
                           shape=dict(dim=[1,1,225,225]))
  else:
    n.data, n.label = L.Python(module = 'data_layer', layer = 'DataLayer',
                               ntop = 2, phase = Train_Mode[mode],
                               param_str = str(data_params))
  
  n.conv1_a, n.relu1_a = conv_relu(n.data,15,64,3,param=param,name_prefix='conv1_a')
  n.pool1_a = pooling(n.relu1_a,3, 2)
  
  n.conv2_a, n.relu2_a = conv_relu(n.pool1_a,5,128,param=param,name_prefix='conv2_a')
  n.pool2_a = pooling(n.relu2_a,3,2)
  
  n.conv3_a, n.relu3_a = conv_relu(n.pool2_a,3,256,param=param,name_prefix='conv3_a')
  n.conv4_s, n.relu4_s = conv_relu(n.relu3_a,3,256,param=param,name_prefix='conv4_s')
  
  n.conv5_s, n.relu5_s = conv_relu(n.relu4_s,3,256,param=param,name_prefix='conv5_s')
  n.pool5_s = pooling(n.relu5_s,3,2)
  
  n.fc6_s, n.relu6_s = fc_relu(n.pool5_s, 512,param=param,name_prefix='fc6_s')
  
  if mode == 'train':
    n.drop6_s = fc7input = L.Dropout(n.relu6_s, dropout_ratio=0.55,in_place=True)
  else:
    fc7input = n.relu6_s;
  n.fc7_s, n.relu7_s = fc_relu(fc7input, 512, param=param,name_prefix='fc7_s')
  if mode =='train':
    n.drop7_s = fc8input= L.Dropout(n.relu7_s, dropout_ratio = 0.55,in_place=True)
  else:
    fc8input = n.relu7_s
  #n.feat8_r_s = fullconnect(fc8input, 100,param=learned_param,name_prefix='fc8_r_s')
  n.feat8_s = fullconnect(fc8input, num_class,param=learned_param,name_prefix='fc8_s')
  
  if mode != 'deploy':
    n.loss = L.SoftmaxWithLoss(n.feat8_s, n.label)
  
  if mode=='test': #validation
    n.accuracy = L.Accuracy(n.feat8_s, n.label, phase = Train_Mode[mode])
  
  proto = n.to_proto()
  proto.name = 'SketchANet'
  return proto

def pretrain_image(data_params, num_class = 20, mode = 'train',learn_all=True):
  """ our version of Sketch-A-Net
  data_params: batch_size, source, shape, scale, rot
  val_mode: 0 if this is train net, 1 if test net, 2 if deploy net
  """
  param = learned_param if learn_all else frozen_param
  n = caffe.NetSpec()
  if mode == 'deploy':
    n.data        = L.Input(name='data',
                           shape=dict(dim=[1,1,225,225]))
  else:
    n.data, n.label = L.Python(module = 'data_layer', layer = 'DataLayer',
                               ntop = 2, phase = Train_Mode[mode],
                               param_str = str(data_params))
  
  n.conv1_p, n.relu1_p = conv_relu(n.data,15,64,3,param=param,name_prefix='conv1_p')
  n.pool1_p = pooling(n.relu1_p,3, 2)
  
  n.conv2_p, n.relu2_p = conv_relu(n.pool1_p,5,128,param=param,name_prefix='conv2_p')
  n.pool2_p = pooling(n.relu2_p,3,2)
  
  n.conv3_p, n.relu3_p = conv_relu(n.pool2_p,3,256,param=param,name_prefix='conv3_p')
  n.conv4, n.relu4 = conv_relu(n.relu3_p,3,256,param=param,name_prefix='conv4')
  
  n.conv5, n.relu5 = conv_relu(n.relu4,3,256,param=param,name_prefix='conv5')
  n.pool5 = pooling(n.relu5,3,2)
  
  n.fc6, n.relu6 = fc_relu(n.pool5, 512,param=param,name_prefix='fc6')
  
  if mode == 'train':
    n.drop6 = fc7input = L.Dropout(n.relu6, dropout_ratio=0.55,in_place=True)
  else:
    fc7input = n.relu6;
  n.fc7, n.relu7 = fc_relu(fc7input, 512, param=param,name_prefix='fc7')
  if mode =='train':
    n.drop7 = fc8input= L.Dropout(n.relu7, dropout_ratio = 0.55,in_place=True)
  else:
    fc8input = n.relu7
  #n.feat8_r = fullconnect(fc8input, 100,param=learned_param,name_prefix='fc8_r')
  n.feat8 = fullconnect(fc8input, num_class,param=learned_param,name_prefix='fc8')
  
  if mode != 'deploy':
    n.loss = L.SoftmaxWithLoss(n.feat8, n.label)
  
  if mode=='test': #validation
    n.accuracy = L.Accuracy(n.feat8, n.label, phase = Train_Mode[mode])
  
  proto = n.to_proto()
  proto.name = 'SketchANet'
  return proto

def double_branch(data_params, loss_params, num_class = 20, phase = 'train',\
                   share_weights='half',loss_type='old'):
    """define the two-branch network consisting sketch branch and image (edge)
    branch
    both branchs have softmax loss"""
    param0 = frozen_param
    param = learned_param
    cn = '_p'   #common name added to prefix, applied to pos and neg branch
    if share_weights == 'full':
      an_t = an_b = cn   #anchor name for top and bottom layers
    elif share_weights == 'half':
      an_t = '_a'
      an_b = cn
      print('half-share selected')
    else: #no sharing weights
      an_t = an_b = '_a'
    
    print 'double_branch selected'
    print('top anchor: {}, bottom anchor: {}'.format(an_t,an_b))
    n = caffe.NetSpec()
    n.data_s, n.data_p, n.label_s,n.label_i = L.Python(name = 'input_data',
                                   ntop=4, module='data_layer_2branchs',layer = 'DataLayer',
                                   param_str = str(data_params))
    
    #sketch branch
    n.conv1_a, n.relu1_a = conv_relu(n.data_s,15,64,3,param=param0,name_prefix='conv1'+an_t)
    n.pool1_a = pooling(n.relu1_a,3, 2)
    
    n.conv2_a, n.relu2_a = conv_relu(n.pool1_a,5,128,param=param0,name_prefix='conv2'+an_t)
    n.pool2_a = pooling(n.relu2_a,3,2)
    
    n.conv3_a, n.relu3_a = conv_relu(n.pool2_a,3,256,param=param0,name_prefix='conv3'+an_t)
    n.conv4_a, n.relu4_a = conv_relu(n.relu3_a,3,256,param=param,name_prefix='conv4'+an_b)
    
    n.conv5_a, n.relu5_a = conv_relu(n.relu4_a,3,256,param=param,name_prefix='conv5'+an_b)
    n.pool5_a = pooling(n.relu5_a,3,2)
    
    n.fc6_a, n.relu6_a = fc_relu(n.pool5_a, 512,param=param,name_prefix='fc6'+an_b)
    n.drop6_a = L.Dropout(n.relu6_a, dropout_ratio=0.55,in_place=True)
    n.fc7_a, n.relu7_a = fc_relu(n.drop6_a, 512, param=param,name_prefix='fc7'+an_b)
    n.drop7_a = L.Dropout(n.relu7_a, dropout_ratio = 0.55,in_place=True)
    n.feat_a = fullconnect(n.drop7_a, 100,param=learned_param,name_prefix='fc8'+an_b)
    n.feat8_a = fullconnect(n.feat_a, num_class,param=learned_param,name_prefix='clas_p')
    n.loss_a = L.SoftmaxWithLoss(n.feat8_a, n.label_s)
    
    #image branch
    n.conv1_p, n.relu1_p = conv_relu(n.data_p,15,64,3,param=param0,name_prefix='conv1'+cn)
    n.pool1_p = pooling(n.relu1_p,3, 2)
    
    n.conv2_p, n.relu2_p = conv_relu(n.pool1_p,5,128,param=param0,name_prefix='conv2'+cn)
    n.pool2_p = pooling(n.relu2_p,3,2)
    
    n.conv3_p, n.relu3_p = conv_relu(n.pool2_p,3,256,param=param0,name_prefix='conv3'+cn)
    n.conv4_p, n.relu4_p = conv_relu(n.relu3_p,3,256,param=param,name_prefix='conv4'+cn)
    
    n.conv5_p, n.relu5_p = conv_relu(n.relu4_p,3,256,param=param,name_prefix='conv5'+cn)
    n.pool5_p = pooling(n.relu5_p,3,2)
    
    n.fc6_p, n.relu6_p = fc_relu(n.pool5_p, 512,param=param,name_prefix='fc6'+cn)
    n.drop6_p = L.Dropout(n.relu6_p, dropout_ratio=0.55,in_place=True)
    n.fc7_p, n.relu7_p = fc_relu(n.drop6_p, 512, param=param,name_prefix='fc7'+cn)
    n.drop7_p = L.Dropout(n.relu7_p, dropout_ratio = 0.55,in_place=True)
    n.feat_p = fullconnect(n.drop7_p, 100,param=learned_param,name_prefix='fc8'+cn)
    n.feat8_p = fullconnect(n.feat_p, num_class,param=learned_param,name_prefix='clas_p')
    n.loss_p = L.SoftmaxWithLoss(n.feat8_p, n.label_i,loss_weight = 2.0)
    
    proto = n.to_proto()
    proto.name = 'SketchANet'
    return proto
    
def double_branch_plus(data_params, loss_params, num_class = 20, phase = 'train',\
                   share_weights='half',loss_type='old'):
    """ same as double_branch but add contrastive loss
    define the two-branch network consisting sketch branch and image (edge)
    branch
    both branchs have softmax loss"""
    param0 = frozen_param
    param = learned_param
    cn = '_p'   #common name added to prefix, applied to pos and neg branch
    if share_weights == 'full':
      an_t = an_b = cn   #anchor name for top and bottom layers
    elif share_weights == 'half':
      an_t = '_a'
      an_b = cn
      print('half-share selected')
    else: #no sharing weights
      an_t = an_b = '_a'
    
    print 'double_branch selected'
    print('top anchor: {}, bottom anchor: {}'.format(an_t,an_b))
    n = caffe.NetSpec()
    n.data_s, n.data_p, n.label_s,n.label_i = L.Python(name = 'input_data',
                                     ntop=4, module='data_layer_siamese',layer = 'DataLayer',
                                     param_str = str(data_params))
    
    #sketch branch
    n.conv1_a, n.relu1_a = conv_relu(n.data_s,15,64,3,param=param0,name_prefix='conv1'+an_t)
    n.pool1_a = pooling(n.relu1_a,3, 2)
    
    n.conv2_a, n.relu2_a = conv_relu(n.pool1_a,5,128,param=param0,name_prefix='conv2'+an_t)
    n.pool2_a = pooling(n.relu2_a,3,2)
    
    n.conv3_a, n.relu3_a = conv_relu(n.pool2_a,3,256,param=param0,name_prefix='conv3'+an_t)
    n.conv4_a, n.relu4_a = conv_relu(n.relu3_a,3,256,param=param,name_prefix='conv4'+an_b)
    
    n.conv5_a, n.relu5_a = conv_relu(n.relu4_a,3,256,param=param,name_prefix='conv5'+an_b)
    n.pool5_a = pooling(n.relu5_a,3,2)
    
    n.fc6_a, n.relu6_a = fc_relu(n.pool5_a, 512,param=param,name_prefix='fc6'+an_b)
    n.drop6_a = L.Dropout(n.relu6_a, dropout_ratio=0.55,in_place=True)
    n.fc7_a, n.relu7_a = fc_relu(n.drop6_a, 512, param=param,name_prefix='fc7'+an_b)
    n.drop7_a = L.Dropout(n.relu7_a, dropout_ratio = 0.55,in_place=True)
    n.feat_a = fullconnect(n.drop7_a, 100,param=learned_param,name_prefix='fc8'+an_b)
    n.feat8_a = fullconnect(n.feat_a, num_class,param=learned_param,name_prefix='clas_p')
    n.loss_a = L.SoftmaxWithLoss(n.feat8_a, n.label_s)
    
    #image branch
    n.conv1_p, n.relu1_p = conv_relu(n.data_p,15,64,3,param=param0,name_prefix='conv1'+cn)
    n.pool1_p = pooling(n.relu1_p,3, 2)
    
    n.conv2_p, n.relu2_p = conv_relu(n.pool1_p,5,128,param=param0,name_prefix='conv2'+cn)
    n.pool2_p = pooling(n.relu2_p,3,2)
    
    n.conv3_p, n.relu3_p = conv_relu(n.pool2_p,3,256,param=param0,name_prefix='conv3'+cn)
    n.conv4_p, n.relu4_p = conv_relu(n.relu3_p,3,256,param=param,name_prefix='conv4'+cn)
    
    n.conv5_p, n.relu5_p = conv_relu(n.relu4_p,3,256,param=param,name_prefix='conv5'+cn)
    n.pool5_p = pooling(n.relu5_p,3,2)
    
    n.fc6_p, n.relu6_p = fc_relu(n.pool5_p, 512,param=param,name_prefix='fc6'+cn)
    n.drop6_p = L.Dropout(n.relu6_p, dropout_ratio=0.55,in_place=True)
    n.fc7_p, n.relu7_p = fc_relu(n.drop6_p, 512, param=param,name_prefix='fc7'+cn)
    n.drop7_p = L.Dropout(n.relu7_p, dropout_ratio = 0.55,in_place=True)
    n.feat_p = fullconnect(n.drop7_p, 100,param=learned_param,name_prefix='fc8'+cn)
    n.feat8_p = fullconnect(n.feat_p, num_class,param=learned_param,name_prefix='clas_p')
    n.loss_p = L.SoftmaxWithLoss(n.feat8_p, n.label_i,loss_weight = 2.0)
    
    layer = 'ContrastiveLossLayer' + str(loss_type)
    n.loss    = L.Python(n.feat_a, n.feat_p, n.label_s,n.label_i, 
                          name = 'contrastive_loss', loss_weight = loss_params['loss_weight'], 
                          module = 'contrastive_loss', layer = layer, 
                          param_str = str(loss_params))
    
    proto = n.to_proto()
    proto.name = 'SketchANet'
    return proto
    
def siamese_anchor(out_dim=100):
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
  #n.norm_a = L.Normalize(n.feat_a,in_place=True)
  
  proto = n.to_proto()
  proto.name = 'SketchTriplet'
  return proto
  
def siamese_pos(out_dim=100):
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
  #n.norm_p = L.Normalize(n.feat_p,in_place=True)
  
  proto = n.to_proto()
  proto.name = 'SketchTriplet'
  return proto

def siamese(data_params, loss_params, num_class = 20, phase = 'train',\
                   share_weights='full',loss_type=0):
    """define the two-branch network consisting sketch branch and image (edge)
    branch
    both branchs have softmax loss"""
    param0 = frozen_param
    param = learned_param
    cn = '_p'   #common name added to prefix, applied to pos and neg branch
    if share_weights == 'full':
      an_t = an_b = cn   #anchor name for top and bottom layers
    elif share_weights == 'half':
      an_t = '_a'
      an_b = cn
      print('half-share selected')
    else: #no sharing weights
      an_t = an_b = '_a'
    if phase == 'train' or phase == 'test':
      print 'double_branch selected'
      print('top anchor: {}, bottom anchor: {}'.format(an_t,an_b))
      n = caffe.NetSpec()
      n.data_s, n.data_p, n.label_s,n.label_i = L.Python(name = 'input_data',
                                     ntop=4, module='data_layer_siamese',layer = 'DataLayer',
                                     param_str = str(data_params))
      
      #sketch branch
      n.conv1_a, n.relu1_a = conv_relu(n.data_s,15,64,3,param=param,name_prefix='conv1'+an_t)
      n.pool1_a = pooling(n.relu1_a,3, 2)
      
      n.conv2_a, n.relu2_a = conv_relu(n.pool1_a,5,128,param=param,name_prefix='conv2'+an_t)
      n.pool2_a = pooling(n.relu2_a,3,2)
      
      n.conv3_a, n.relu3_a = conv_relu(n.pool2_a,3,256,param=param,name_prefix='conv3'+an_t)
      n.conv4_a, n.relu4_a = conv_relu(n.relu3_a,3,256,param=param,name_prefix='conv4'+an_b)
      
      n.conv5_a, n.relu5_a = conv_relu(n.relu4_a,3,256,param=param,name_prefix='conv5'+an_b)
      n.pool5_a = pooling(n.relu5_a,3,2)
      
      n.fc6_a, n.relu6_a = fc_relu(n.pool5_a, 512,param=param,name_prefix='fc6'+an_b)
      n.drop6_a = L.Dropout(n.relu6_a, dropout_ratio=0.55,in_place=True)
      n.fc7_a, n.relu7_a = fc_relu(n.drop6_a, 512, param=param,name_prefix='fc7'+an_b)
      n.drop7_a = L.Dropout(n.relu7_a, dropout_ratio = 0.55,in_place=True)
      #n.fc8_a, n.feat_a     = fc_norm_triplet(n.drop7_a, 100, 'fc8'+an_b)
      n.feat_a = fullconnect(n.drop7_a, 100,param=learned_param,name_prefix='fc8'+an_b)
  #==============================================================================
  #     n.feat8_a = fullconnect(n.feat_a, num_class,param=learned_param,name_prefix='clas_p')
  #     n.loss_a = L.SoftmaxWithLoss(n.feat8_a, n.label_s)
  #==============================================================================
      
      #image branch
      n.conv1_p, n.relu1_p = conv_relu(n.data_p,15,64,3,param=param,name_prefix='conv1'+cn)
      n.pool1_p = pooling(n.relu1_p,3, 2)
      
      n.conv2_p, n.relu2_p = conv_relu(n.pool1_p,5,128,param=param,name_prefix='conv2'+cn)
      n.pool2_p = pooling(n.relu2_p,3,2)
      
      n.conv3_p, n.relu3_p = conv_relu(n.pool2_p,3,256,param=param,name_prefix='conv3'+cn)
      n.conv4_p, n.relu4_p = conv_relu(n.relu3_p,3,256,param=param,name_prefix='conv4'+cn)
      
      n.conv5_p, n.relu5_p = conv_relu(n.relu4_p,3,256,param=param,name_prefix='conv5'+cn)
      n.pool5_p = pooling(n.relu5_p,3,2)
      
      n.fc6_p, n.relu6_p = fc_relu(n.pool5_p, 512,param=param,name_prefix='fc6'+cn)
      n.drop6_p = L.Dropout(n.relu6_p, dropout_ratio=0.55,in_place=True)
      n.fc7_p, n.relu7_p = fc_relu(n.drop6_p, 512, param=param,name_prefix='fc7'+cn)
      n.drop7_p = L.Dropout(n.relu7_p, dropout_ratio = 0.55,in_place=True)
      #n.fc8_p, n.feat_p     = fc_norm_triplet(n.drop7_p, 100, 'fc8'+cn)
      n.feat_p = fullconnect(n.drop7_p, 100,param=learned_param,name_prefix='fc8'+cn)
  #==============================================================================
  #     n.feat8_p = fullconnect(n.feat_p, num_class,param=learned_param,name_prefix='clas_p')
  #     n.loss_p = L.SoftmaxWithLoss(n.feat8_p, n.label_i,loss_weight = 2.0)
  #==============================================================================
      
      #contrastive loss
      layer = 'ContrastiveLossLayer' + str(loss_type)
      n.loss    = L.Python(n.feat_a, n.feat_p, n.label_s,n.label_i, 
                            name = 'contrastive_loss', loss_weight = loss_params['loss_weight'], 
                            module = 'contrastive_loss', layer = layer, 
                            param_str = str(loss_params))
      proto = n.to_proto()
      proto.name = 'SketchANet'
      return proto
    else:
      if share_weights=='full':
        return siamese_anchor(100)
      else:
        return siamese_anchor(100),siamese_pos(100)

if __name__ == '__main__':
    make_net()