# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:09:53 2016
@author: tb00083
extract triplet cnn features
INPUT: deploy.prototxt, weights and input image or sketch
OUTPUT: extracted features (.npz or .mat)
"""

CAFFE_DEVICE = 0  #GPU device to be used
layer=0           #last layer in prototxt

domain = 'image' # 'image' or 'sketch'
INPUT = 'sample_image.jpg'
MEAN_DB = 'models/image_mean.binaryproto'
WEIGHTS = 'models/triplet1_250c80s100i_iter_1000000.caffemodel'


"""import libraries"""
import sys
sys.path.insert(1,'./Utils')
import caffe
from caffe_func_utils import biproto2py
import numpy as np
from PIL import Image
from skimage import feature
from bwmorph import bwmorph_thin

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
  if domain == 'image':
    DEPLOY = 'models/deploy_images.prototxt'
    scale_factor = 1.0
  else:
    DEPLOY = 'models/deploy_sketch.prototxt'
    scale_factor = 2.0
  
  net = get_net(WEIGHTS, DEPLOY)
  mean_image = biproto2py(MEAN_DB).squeeze()
  
  """pre-processing"""
  img = preprocess(INPUT, mean_image, domain)
  
  """feat extraction"""
  net.blobs[net.inputs[0]].reshape(1,1,225,225)
  feat = extractitem(net, img) * scale_factor
  
  print 'Done.'