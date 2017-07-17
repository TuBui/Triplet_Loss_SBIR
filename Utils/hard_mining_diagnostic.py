#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:45:46 2017
class to view images
@author: tb00083
"""

import numpy as np
from PIL import Image
from helper import helper
from caffe_class_utils import lmdbs
import os
class ImagePack(object):
  def __init__(self,img_dir='',img_lst='',lmdb=''):
    self.have_img_lst = self.have_img_dir = self.have_lmdb = False
    self.hardsel = None
    if img_lst:
      helps = helper()
      self.img_lst = helps.read_list(img_lst,delimeter = ',',keep_original = False)
      self.have_img_lst = True
    if img_dir:
      assert os.path.isdir(img_dir), 'Opps. img_dir is not a dir'
      self.img_dir = img_dir
      self.have_img_dir = True
    if lmdb:
      self.lmdb = lmdbs(lmdb)
      self.have_lmdb = True
  
  def get_images(self,ids,ver_con = False):
    """given ids get images and concatenate them into a single image"""
    inds = np.array(ids)
    #inds = ids
    out = {'img': None,'lmdb':None,'label_id':None,'path': None}
    if self.have_lmdb:
#==============================================================================
#       if inds.size ==1:
#         tmp = self.lmdb.get_datum(int(inds))
#       else:
#         tmp = self.lmdb.get_data(inds)
#==============================================================================
      if ver_con:
        tmp = self.lmdb.get_data(inds).transpose(0,2,3,1)
        out['lmdb'] = tmp.reshape(tmp.shape[0]*tmp.shape[1],tmp.shape[2],tmp.shape[3])\
          .squeeze().astype(np.uint8)
      else:
        tmp = self.lmdb.get_data(inds).transpose(0,3,2,1)
        out['lmdb'] = tmp.reshape(tmp.shape[0]*tmp.shape[1],tmp.shape[2],tmp.shape[3])\
          .transpose(1,0,2).squeeze().astype(np.uint8)
    
    if self.have_img_lst:
      out['label_id'] = self.img_lst[0][inds]
      out['path'] = [self.img_lst[1][i] for i in inds]
      if self.have_img_dir:
        out['img'] = [np.array(Image.open(os.path.join(self.img_dir,self.img_lst[1][i]))\
                    .convert('RGB').resize((256,256))) for i in inds]
        if ver_con:
          out['img'] = np.concatenate(out['img'],axis=0)
        else:
          out['img'] = np.concatenate(out['img'],axis=1)

    return out

class HardMining(object):
  def __init__(self):
    self.img_data = self.skt_data = None
    self.hardsel_res = None
    
  def RegImage(self,img_dir,img_lst,img_lmdb):
    self.img_data = ImagePack(img_dir=img_dir,img_lst=img_lst,lmdb=img_lmdb)
    
  def RegSketch(self,skt_dir,skt_lst,skt_lmdb):
    self.skt_data = ImagePack(img_dir=skt_dir,img_lst=skt_lst,lmdb=skt_lmdb)
  
  def RegHardsel(self,hardsel_file):
    self.hardsel_res = None
    self.hardsel_res = np.load(hardsel_file)
  
  def GetHardNeg(self,ind):
    """ind: index of the query, singular"""
    assert isinstance(ind,(int,long)),'Opps. ind is not a singular'
    skt = self.skt_data.get_images([ind,])
    imgs = self.img_data.get_images(self.hardsel_res['neg'][ind])
    out = {}
    pad = np.zeros((256,5,3),dtype=np.uint8)
    #lmdb concat
    out['lmdb'] = np.concatenate((to_RGB(skt['lmdb']),pad,to_RGB(imgs['lmdb'])),axis=1)
    #src concat
    out['src'] = np.concatenate((to_RGB(skt['img']),pad,to_RGB(imgs['img'])),axis=1)
    #meta
    out['aux'] = dict(path = (skt['path'],imgs['path']),\
                    label_id = (skt['label_id'],imgs['label_id']))
    return out
  
  def GetHardPos(self,ind):
    """ind: index of the query, singular"""
    assert isinstance(ind,(int,long)),'Opps. ind is not a singular'
    skt = self.skt_data.get_images([ind,])
    imgs = self.img_data.get_images(self.hardsel_res['pos'][ind])
    out = {}
    pad = np.zeros((256,5,3),dtype=np.uint8)
    #lmdb concat
    out['lmdb'] = np.concatenate((to_RGB(skt['lmdb']),pad,to_RGB(imgs['lmdb'])),axis=1)
    #src concat
    out['src'] = np.concatenate((to_RGB(skt['img']),pad,to_RGB(imgs['img'])),axis=1)
    #meta
    out['aux'] = dict(path = (skt['path'],imgs['path']),\
                    label_id = (skt['label_id'],imgs['label_id']))
    return out
    
  def GetGeneral(self,ids_q,ids_d):
    """return sbir images in general
    ids_q: id of query
    ids_d: id of returned images with row: query, collumn: rank
    ids_d should be extracted from function_class_util.py: sbir_hardsel.retrieve()
    num: number of images to display"""
    ids_q_ = [ids_q,] if isinstance(ids_q,(int,long)) else ids_q
    if len(ids_d.shape)==1:
      ids_d = ids_d[None,...] 
    skt = self.skt_data.get_images(ids_q_,ver_con = True)
    img_data = [self.img_data.get_images(ids) for ids in ids_d]
    img_src = np.concatenate([img['img'] for img in img_data],axis=0)
    img_lmdb = np.concatenate([img['lmdb'] for img in img_data],axis=0)
    
    #convert 2RGB
    skt['lmdb'] = to_RGB(skt['lmdb'])
    skt['img'] = to_RGB(skt['img'])
    img_lmdb = to_RGB(img_lmdb)

    pad = np.zeros((img_src.shape[0],5,3),dtype=np.uint8)
    #print 'skt: ',skt['lmdb'].shape, ' pad: ', pad.shape,' img: ',img_lmdb.shape 
    out = {}
    out['lmdb'] = np.concatenate((skt['lmdb'],pad,img_lmdb),axis = 1)
    out['src'] = np.concatenate((skt['img'],pad,img_src),axis = 1)
    out['aux'] = dict(path=(skt['path'],[img['path'] for img in img_data]), \
                      label = (skt['label_id'],[img['label_id'] for img in img_data]))
    
    return out
    
def to_RGB(im):
  if len(im.shape) ==2:
    return im[...,None].repeat(3,2)
  elif len(im.shape) == 3:
    return im
  else:
    assert False,'Opps. Unable to recognise image dimension'
    
if __name__ == '__main__':
  img_dir = '/vol/vssp/ddascratch/Tu/classification/cnn/PTSC/DATA/clas6_leonardo/seg_clas_ret/datasets/images_250c100s'
  img_lst = '/vol/vssp/ddascratch/Tu/classification/pycaffe/CVIU16/db/image80/list.txt'
  lmdb = '/vol/vssp/ddascratch/Tu/classification/pycaffe/CVIU16/db/image80'
  
  img_pack = ImagePack(img_dir = img_dir, img_lst = img_lst, lmdb = lmdb)
  tmp = img_pack.get_images([0,2,3])
  print('Done')