# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:53:05 2016

@author: tb00083
"""
import sys

def addpath(server_name="rollo",disp=1):
  if server_name == "nimloth":
    pycaffe_path = "/vol/vssp/ddawrk/Tu/Toolkits/caffe/caffe_nimloth/caffe/python"
  else:
    pycaffe_path = "/vol/vssp/ddawrk/Tu/Toolkits/caffe/caffe_rollo/caffe/python"
  
  if pycaffe_path not in sys.path:
    if disp:
      print "Added pycaffe path: " + pycaffe_path
    sys.path.insert(0,pycaffe_path)
  
  pycaffe_utils = "/vol/vssp/ddawrk/Tu/code/classification/pycaffe/Utils"
  if pycaffe_utils not in sys.path:
    if disp:
      print "Added " + pycaffe_utils
    sys.path.insert(1,pycaffe_utils)

if __name__ == '__main__':
  addpath()