# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:45:06 2016

@author: tb00083
"""

import numpy as np
from scipy import ndimage
from numpy import matrix as mat

class SimpleAugment:

  """
  SimpleTransformer is a simple class for preprocessing and deprocessing
  images for caffe.
  currently support image 2-D only, i.e. len(im.shape)=2
  """

  def __init__(self, mean=0, scale=1.0, shape = (225,225), rot = 0):
    self.mean = np.array(mean, dtype=np.float32)
    self.scale = np.float32(scale)
    self.outshape = shape
    self.rot = rot

  def set_mean(self, mean):
    """
    Set the mean to subtract for centering the data.
    """
    self.mean = mean
      
  def set_shape(self, shape):
    self.outshape = shape

  def set_scale(self, scale):
    """
    Set the data scaling.
    """
    self.scale = scale
  
  def set_rotation(self, rot):
    """
    Set the rotation range (in degree)
    """
    self.rot = rot

  def preprocess(self, im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """
    
    im = np.float32(im)
    im = im[:, :, ::-1]  # change to BGR
    im -= self.mean
    im *= self.scale
    im = im.transpose((2, 0, 1))

    return im

  def deprocess(self, im):
    """
    inverse of preprocess()
    """
    im = im.transpose(1, 2, 0)
    im /= self.scale
    im += self.mean
    im = im[:, :, ::-1]  # change to RGB

    return np.uint8(im)
      
  def augment(self,im):
    """
    augmentation includes: mean subtract, random crop, random mirror, random rotation
    """
    #random rotation
    if self.rot != 0:
      rot_ang = np.random.randint(self.rot[0], high = self.rot[1]+1)
      im = ndimage.rotate(im.astype(np.uint8), rot_ang, \
                          cval=255,reshape = False)
    
    #mean subtract and scaling
    im = np.float32(im)
    im -= self.mean
    im *= self.scale
    
    #random flip
    flip = np.random.choice(2)*2-1
    im = im[:, ::flip]
    
    #random crop
    y_offset = np.random.randint(im.shape[0] - self.outshape[0] - 8)
    x_offset = np.random.randint(im.shape[1] - self.outshape[1] - 8)
    return im[4+y_offset:4+self.outshape[0]+y_offset,
              4+x_offset:4+self.outshape[1]+x_offset]
    #return im
  
  def augment_deploy(self,ims):
    """
    same as augment() but applied to deploy network only:
    action: subtract mean, apply scale, central crop
    """
    ims = np.float32(ims)
    ims -= self.mean
    ims *= self.scale
    x_offset = (ims.shape[-1] - self.outshape[-1])/2
    y_offset = (ims.shape[-2] - self.outshape[-2])/2
    return ims[...,y_offset:y_offset+self.outshape[-2],
               x_offset:x_offset+self.outshape[-1]]

def edge_rotate(im,ang):
  """
  rotate edge map using nearest neighbour preserving edge and dimension
  Assumption: background 255, foreground 0
  currently does not work as good as ndimage.rotate
  """
  ang_rad = np.pi / 180.0 * ang
  H,W = np.float32(im.shape)
  R = mat([[np.cos(ang_rad),-np.sin(ang_rad) ,0],
        [np.sin(ang_rad), np.cos(ang_rad),0],
        [0              ,0               ,1.0]])
  T0 = mat([[1.0,0,-W/2],[0,1.0,-H/2],[0,0,1.0]])
  M0 = T0.I * R * T0
  
  tl_x,tl_y = np.floor(warping([0,0],M0))
  tr_x,tr_y = np.floor(warping([W-1,0],M0))
  bl_x,bl_y = np.floor(warping([0,H-1],M0))
  br_x,br_y = np.floor(warping([W-1,H-1],M0))
  
  minx = np.min([tl_x,tr_x,bl_x,br_x])
  maxx = np.max([tl_x,tr_x,bl_x,br_x])
  miny = np.min([tl_y,tr_y,bl_y,br_y])
  maxy = np.max([tl_y,tr_y,bl_y,br_y])
  T1 = mat([[1.0,0.0,minx],
            [0.0,1.0,miny],
            [0.0,0.0,1.0]])
  M1 = M0.I * T1
  nW = int(maxx - minx+1)
  nH = int(maxy - miny+1)
  
  out = np.ones((nH,nW),dtype=np.float32)*255
  for y in xrange(nH):
    for x in xrange(nW):
      u,v = np.int64(warping([x,y],M1))
      if u>=0 and u<W and v>=0 and v<H and im[v,u]!=255:
        out[y,x]=0
  
  return out

def warping(p,M):
  """Given 2-D point p and  warping matrix M, find the new location of p
  p:type list
  M: type matrix
  """
  p0 = mat([p[0],p[1],1.0]).T
  out = M*p0
  out /= out[2]
  return out[0],out[1]