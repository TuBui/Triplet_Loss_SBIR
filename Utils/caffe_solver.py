# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:45:46 2016

@author: tb00083
"""
""" assume that caffe is in the search path
"""
from caffe.proto import caffe_pb2
from google.protobuf import text_format

class SolverConfig:

  """
  SolverConfig is a class for creating a solver.prototxt file. It sets default
  values and can export a solver parameter file.
  """

  def __init__(self, solver_path = '', debug=False):
    """
    Initialise solver params
    If a file is given, SolverConfig is initialised with params from that file
    """
    self.sp = caffe_pb2.SolverParameter()
    #critical:
    self.sp.base_lr = 0.01
    self.sp.momentum = 0.9
    
    if solver_path:
      self.read(solver_path)
    
    if debug:
      self.sp.max_iter = 12
      self.sp.display = 1
    
    self.sp.type = 'SGD'
                 
  def read(self, solver_path):
    """
    Reads a caffe solver prototxt file and updates the Caffesolver
    instance parameters.
    """
    with open(solver_path) as f:
      text_format.Merge(str(f.read()), self.sp)
    f.close()

  def write(self, solver_path,verbose=1):
    """
    Export solver parameters to file
    """
    solver_config = text_format.MessageToString(self.sp)
    with open(solver_path, 'w') as f:
      f.write(solver_config)
    f.close()
    if verbose:
      print 'Solver parameters written to ' + solver_path
      
  def add_params(self,params):
    """
    Set or update solver parameters
    """
    paramstr = ''
    for key, val in params.items():
      self.sp.ClearField(key) #reset field
      if isinstance(val,str):     #if val is a string
        paramstr += (key + ': ' + '"' + val + '"' + '\n')
      elif type(val) is list:     #repeatable field
        for it in val:
          paramstr += (key + ': ' + str(it) + '\n')
      elif type(val) == type(True): #boolean type
        if val:
          paramstr += (key + ': true\n')
        else:
          paramstr += (key + ': false\n')
      else:                       #numerical value
        paramstr += (key + ': ' + str(val) + '\n')
    #apply change
    text_format.Merge(paramstr, self.sp)