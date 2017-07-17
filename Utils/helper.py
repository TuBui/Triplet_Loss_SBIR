# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:05:57 2016
some help functions to perform basic tasks
@author: tb00083
"""
import os,sys
import csv
import numpy as np
import cPickle as pickle

class helper(object):
  """help functions that do basic tasks
  """
  def GetAllFiles(self, dir_path, trim = 0, extension = ''):
    """
    Recursively get list of all files in the given directory
    trim = 1 : trim the dir_path from results, 0 otherwise
    extension: get files with specific format
    """
    file_paths = []  # List which will store all of the full filepaths.
    
    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    
    if trim==1: #trim dir_path from results
      if dir_path[-1] != os.sep:
        dir_path += os.sep
      trim_len = len(dir_path)
      file_paths = [x[trim_len:] for x in file_paths]
    
    if extension: #select only file with specific extension
      extension = extension.lower()
      tlen = len(extension)
      file_paths = [x for x in file_paths if x[-tlen:] == extension]
      
    return file_paths  # Self-explanatory.
    
  def GetAllDirs(self, dir_path, trim = 0):
    """
    Recursively get list of all directories in the given directory
    excluding the '.' and '..' directories
    trim = 1 : trim the dir_path from results, 0 otherwise
    """
    out = []
    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for dirname in directories:
            # Join the two strings in order to form the full filepath.
            dir_full = os.path.join(root, dirname)
            out.append(dir_full)  # Add it to the list.
    
    if trim==1: #trim dir_path from results
      if dir_path[-1] != os.sep:
        dir_path += os.sep
      trim_len = len(dir_path)
      out = [x[trim_len:] for x in out]
      
    return out
  
  def read_list(self, file_path, delimeter = ' ', keep_original = True):
    """
    read list column wise
    """
    out = []
    with open(file_path, 'r') as f:
      reader = csv.reader(f,delimiter=delimeter)
      for row in reader:
          out.append(row)
    out = zip(*out)
    
    if not keep_original and out[0][0].isdigit(): #attempt to convert the 1st column to numerical array
      out[0] = np.array(out[0]).astype(np.int64)
      
    return out
    
  def save(self, file_path, **kwargs):
    """
    save variables to file (using pickle)
    """
    #check if any variable is a dict
    var_count = 0
    for key in kwargs:
      var_count += 1
      if isinstance(kwargs[key],dict):
        sys.stderr.write('Opps! Cannot write a dictionary into pickle')
        sys.exit(1)
    with open(file_path,'wb') as f:
      pickler = pickle.Pickler(f,-1)
      pickler.dump(var_count)
      for key in kwargs:
        pickler.dump(key)
        pickler.dump(kwargs[key])
    
    
  def load(self, file_path, varnum=0):
    """
    load variables that previously saved using self.save()
    varnum : number of variables u want to load (0 mean it will load all)
    Note: if you are loading class instance(s), you must have it defined in advance
    """
    with open(file_path,'rb') as f:
      pickler = pickle.Unpickler(f)
      var_count = pickler.load()
      if varnum:
        var_count = min([var_count,varnum])
      out = {}
      for i in range(var_count):
        key = pickler.load()
        out[key] = pickler.load()
    
    return out