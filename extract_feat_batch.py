# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:09:53 2016
@author: tb00083
test
"""

CAFFE_DEVICE = 0  #GPU device to be used
layer=0           #last layer in prototxt

INPUT = 'sample_image.jpg'
IMG_DIR = '/vol/vssp/AP_datasets/still/SBIRflickr/Flickr15k/images/source'
IMG_LST = '/vol/vssp/ddawrk/Tu/code/reWork/groundtruth2'
SKT_DIR = '/vol/vssp/ddawrk/Tu/sketch/330sketches'
SKT_LST = '/vol/vssp/ddawrk/Tu/code/reWork/lst_sketches'
MEAN_DB = 'models/image_mean.binaryproto'
WEIGHTS = 'models/triplet1_250c80s100i_iter_1000000.caffemodel'
OUT = '/mnt/manfred/scratch/Tu/temp'

"""import libraries"""
import os,sys
sys.path.insert(1,'./Utils')
import caffe
from caffe_func_utils import biproto2py
import numpy as np
from PIL import Image
from skimage import feature
from bwmorph import bwmorph_thin
import pandas as pd
from concurrent import futures

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(CAFFE_DEVICE)
    # load a new model
    return caffe.Net(deploy_file, caffe.TEST, weights = caffemodel)

def preprocess(image_file, mean_image, dm):
  """
  pre-processing image before feeding to the network
  photo image: edge extract
  sketch: skeletonisation
  
  then mean subtraction and centre crop
  """
  try:
    input_image = Image.fromarray(np.uint8(caffe.io.load_image(image_file)*255))#.resize((256,256),Image.BILINEAR)
  except Exception as e:
    print('Error reading {} - {}'.format(image_file, e))
    return np.zeros((1,1,225,225), dtype = np.float32)
  sf = 256.0/max(input_image.size)
  input_image = input_image.resize((int(input_image.width*sf),int(input_image.height*sf)),Image.BILINEAR)
  input_image = np.array(input_image.convert('L')) #grayscale
  if dm == 'image':
    #canny edge extraction
    input_image = feature.canny(input_image, sigma = 3.0)
    input_image = np.float32(255*(1-input_image))
  else:
    #skeletonisation
    input_image = bwmorph_thin(input_image < 250)
    input_image = np.float32(255*(1-input_image))
  #pad
  ph1 = (256 - input_image.shape[0])/2; ph2 = 256 - input_image.shape[0] - ph1
  pw1 = (256 - input_image.shape[1])/2; pw2 = 256 - input_image.shape[1] - pw1
  input_image = np.pad(input_image, ((ph1,ph2),(pw1,pw2)), 'constant', constant_values=255)
  #mean extraction and crop
  input_image -= mean_image
  return input_image[15:240,15:240][None, None,...]

def preprocessWrapper(args):
  """wrapper for preprocess() for multiprocessing"""
  image_file, mean_image, dm = args
  return preprocess(image_file, mean_image, dm)

def extractitem(net, image):
  """
  extract feature
  """
  
  inblob = net.inputs[0]
  outblob = net.blobs.keys()[layer-1]
  net.blobs[inblob].data[...] = image
  _ = net.forward()
  prediction = net.blobs[outblob].data.squeeze()
  return prediction


if __name__ == '__main__':
  #skt domain
  domain_lst = ['sketch', 'image']
  DEPLOY_LST = ['models/deploy_sketch.prototxt', 'models/deploy_images.prototxt']
  scale_factor_lst = [2.0, 1.0]
  LST = [SKT_LST, IMG_LST]
  DIR = [SKT_DIR, IMG_DIR]
  
  mean_image = biproto2py(MEAN_DB).squeeze()
  batch_size = 128
  multiprocessors = futures.ProcessPoolExecutor(8) #open multi processing pool
  for i, domain in enumerate(domain_lst):
    DEPLOY = DEPLOY_LST[i]
    scale_factor = scale_factor_lst[i]
    img_dir = DIR[i]
    img_lst = pd.read_csv(LST[i], sep = ' ', header = None)
    labels = img_lst[0].tolist()
    paths = img_lst[1].tolist()
    
    net = None
    net = get_net(WEIGHTS, DEPLOY)
    
    nimgs = len(labels)
    out = np.empty((nimgs, 100), dtype = np.float32)
    for j in range(0,nimgs,batch_size):
      print('Processing {} #{}/{}'.format(domain, j, nimgs))
      batchIDs = range(j, min(j+batch_size, nimgs))
      batchlen = len(batchIDs)
      net.blobs[net.inputs[0]].reshape(batchlen,1,225,225)
      #pre-processing
      full_paths = [os.path.join(img_dir, paths[id_]) for id_ in batchIDs]
      input_args = zip(full_paths, [mean_image,]*batchlen, [domain,]*batchlen)
      res = multiprocessors.map(preprocessWrapper, input_args)
      preprocessed_imgs = np.concatenate([img for img in res], axis=0)
      
      #feat extraction
      feat = extractitem(net, preprocessed_imgs) * scale_factor
      out[batchIDs,...] = feat
    np.savez(os.path.join(OUT, domain + '.npz'), feat = out, labels = labels)
  multiprocessors.shutdown()
  print 'Done.'