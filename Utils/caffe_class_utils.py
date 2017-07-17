# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:01:37 2016
This module contains several classes that may be useful in caffe
@author: tb00083
"""

import lmdb
import time, os, sys
import numpy as np
import caffe
import PIL.Image
from StringIO import StringIO
import h5py as h5
from helper import helper
from bwmorph import bwmorph_thin
from svg.SVGProcessor import SVGProcessor #for svgs class
import copy
from multiprocessing.pool import ThreadPool

class lmdbpy(object):
  """
  Manipulate LMDB data in caffe
  This class can read/write lmdb, convert matlab imdb to lmdb
  Used mainly when converting matlab imdb to the standard lmdb in caffe.
  To manipulate lmdb data in a caffe layer, use lmdbs instead
  """
  def create_dummy(lmdb_size, num_classes, out_path):
    """ create a dummy lmdb given 4-D size, number of classes and output path
    """
    # delete any previous db files
    try:
        os.remove(out_path + "/data.mdb")
        os.remove(out_path + "/lock.mdb")
        time.sleep(1)
    except OSError:
        pass
    
    start_t = time.clock()
    N = lmdb_size[0]    #number of instances
    # Let's pretend this is interesting data
    X = np.random.rand(N,lmdb_size[1],lmdb_size[2],lmdb_size[3])
    X = np.array(X*255,dtype=np.uint8)
    y = np.int64(num_classes*np.random.rand(N))

    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10
    
    env = lmdb.open(out_path,map_size=map_size)
    txn = env.begin(write=True)
    buffered_in_bytes = 0
    for i in range(N):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = X.shape[1]
      datum.height = X.shape[2]
      datum.width = X.shape[3]
      datum.data = X[i].tobytes() # or .tostring() if numpy < 1.9
      datum.label = int(y[i])
      datum_str = datum.SerializeToString()
      str_id = '{:08}'.format(i)
      
      # The encode is only essential in Python 3
      txn.put(str_id.encode('ascii'), datum_str)
      
      buffered_in_bytes += sys.getsizeof(datum_str)
      
      # flush and generate new transactions if we have more 100mb pending
      if buffered_in_bytes > 100e6:
        buffered_in_bytes = 0
        txn.commit()
        env.sync()
        txn = env.begin(write=True)
    
    # ensure final output was written
    txn.commit()
    env.sync()
    
    # close databases
    env.close()
    end_t = time.clock()
    print "Complete after %d seconds" %(end_t - start_t)

  def write(self, data, label, out_path):
    """write npy image data and label to lmdb
    image data must be 4-D in format num x channels x height x width
    """
    print 'Writing to ' + out_path
    assert data.ndim == 4, 'data must be 4-D format num x channels x height x width'
    try:
        os.remove(out_path + "/data.mdb")
        os.remove(out_path + "/lock.mdb")
        time.sleep(1)
    except OSError:
        pass
    
    start_t = time.time()
    if np.max(data[...]) > 1.0:
      data = data.astype(np.uint8)
    else:
      data = np.array(data*255, dtype=np.uint8)
    
    N = np.size(data,axis=0)
    map_size = data.nbytes *2
    env = lmdb.open(out_path,map_size=map_size)
    txn = env.begin(write=True)
    buffered_in_bytes = 0
    for i in range(N):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = data.shape[1]
      datum.height = data.shape[2]
      datum.width = data.shape[3]
      datum.data = data[i].tobytes() # or .tostring() if numpy < 1.9
      datum.label = int(label[i])
      datum_str = datum.SerializeToString()
      str_id = '{:08}'.format(i)
      
      # The encode is only essential in Python 3
      txn.put(str_id.encode('ascii'), datum_str)
      
      buffered_in_bytes += sys.getsizeof(datum_str)
      
      # flush and generate new transactions if we have more 100mb pending
      if buffered_in_bytes > 100e6:
        buffered_in_bytes = 0
        txn.commit()
        env.sync()
        txn = env.begin(write=True)
    
    # ensure final output was written
    txn.commit()
    env.sync()
    
    # close databases
    env.close()
    end_t = time.time()
    print "Complete after %d seconds" %int(end_t - start_t)
  
  
  def mat2lmdb(self, mat_file, lmdb_path):
    """
    convert imdb mat to lmdb and save it in caffe
    """
    mat_ = mat2py()
    data, label = mat_.read_imdb(mat_file)
    #loaded data has format height x width x num
    #need to convert to num x channels x height x width
    data = data.transpose(2,0,1)
    data = data[:,None, :, :]
    self.write(data, label, lmdb_path)
  
  
  def read(self, in_path):
    """
    read lmdb, return image data and label
    """
    print 'Reading ' + in_path
    env = lmdb.open(in_path, readonly=True)
    N = env.stat()['entries']
    txn = env.begin()
    for i in range(N):
      str_id = '{:08}'.format(i)
      raw_datum = txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      feature = caffe.io.datum_to_array(datum)
      if i==0:
        data = np.zeros((N,feature.shape[0],feature.shape[1],
                         feature.shape[2]),dtype=np.uint8)
        label = np.zeros(N,dtype=np.int64)
      data[i] = feature
      label[i] = datum.label
    env.close()
    return data, label
    
  def read_encoded(self, in_path):
    """
    read lmdb of encoded data, e.g. PNG or JPG in nvidia digit
    """
    env = lmdb.open(in_path, readonly=True)
    N = env.stat()['entries']
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(value)
      s = StringIO()
      s.write(datum.data)
      s.seek(0)
      img = PIL.Image.open(s)
      if count == 0:
        data = np.zeros((N,datum.channels,datum.height, datum.width),dtype=np.uint8)
        label = np.zeros(N,dtype=np.int64)
      data[count] = np.array(img)
      label[count] = datum.label
      count += 1
      
    env.close()
    return data, label

###############################################################################
class lmdbs(object):
  """
  proper lmdb class to read lmdb data
  Recommend to be used in caffe data layer
  """
  def __init__(self,lmdb_path):
    if not os.path.isdir(lmdb_path):
      assert 0,'Invalid lmdb {}\n'.format(lmdb_path)
    self.env = lmdb.open(lmdb_path, readonly=True)
    self.NumDatum = self.env.stat()['entries']
    self.txn = self.env.begin()
  def __del__(self):
    self.env.close()
    
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    labels = np.zeros(self.NumDatum,dtype=np.int64)
    for i in range(self.NumDatum):
      str_id = '{:08}'.format(i)
      raw_datum = self.txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      labels[i] = datum.label
    return labels
  
  def get_datum(self,ind):
    """get datum in lmdb given its index"""
    str_id = '{:08}'.format(ind)
    raw_datum = self.txn.get(str_id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    img = caffe.io.datum_to_array(datum)
    return img
  
  def get_data(self,inds):
    """
    get array of data given its indices
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i])
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data
  
  def extract(self, lmdb_path,ids=0):
    """extract particular image(s) from lmdb (of non-encoded data)
    Deprecated. Use get_data instead
    """
    ids = np.array(ids)
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin()
    for i in range(ids.size):
      str_id = '{:08}'.format(ids[i])
      raw_datum = txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      feature = caffe.io.datum_to_array(datum)
      if i==0:
        X = np.zeros((len(ids),feature.shape[0],feature.shape[1],
                      feature.shape[2]),dtype=np.uint8)
        Y = np.zeros(len(ids),dtype=np.int64)
      X[i] = feature
      Y[i] = datum.label
    env.close()
    return (X,Y)

###############################################################################
class svgs(object):
  """
  a class to read svg data
  Recommend to be used in caffe data layer
  """
  def __init__(self,lmdb_path):
    if not os.path.isfile(lmdb_path):
      assert 0,'Invalid .pkl lmdb {}\n'.format(lmdb_path)
    
    #note: we need to define svg class prior to loading it
    helps = helper()
    self.svgdb = helps.load(lmdb_path)
    
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    return self.svgdb['labels'].astype(np.int64)
  
  def get_datum(self,ind):
    """get datum in lmdb given its index
    Preprocess:  randomly remove strokes and skeletonise
    """
    #svg = copy.deepcopy(self.svgdb['data'][ind])
    svg = self.svgdb['data'][ind]
#==============================================================================
#     svg.randomly_occlude_strokes(n=4)
#     img = svg.get_numpy_array(size=256)
#==============================================================================
    
    img = svg.remove_strokes_and_convert(n=4,size=256)
    #convert to grayscale binary
    img = (0.299*img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]) < 254
    img = bwmorph_thin(img)
    img = np.float32(255*(1-img))
    #svg = None
    return img
  
  def get_data(self,inds):
    """
    get array of data given its indices
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i])
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data
  
  def get_num_strokes(self):
    """
    special function to get list of stroke number of the svg database
    """
    out = [svg.get_num_strokes() for svg in self.svgdb['data']]
    return out

###############################################################################
class mat2py(object):
  """Manipulate matlab v7.3 (.mat) file
  Most of the case you need to know in advance the structure of saved variables
  in the .mat file.
  there is an odd problem that prevents caffe from working if data is read 
   from mat file. So this class is deprecated until the problem is solved
  """
  
  def read_imdb(self,mat_file):
    """
    mat file variables:
    images
      data          #
      labels        #
      data_mean
    meta
      classes
    """
    assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
    f = h5.File(mat_file)
    data = np.array(f['images']['data'])
    labels = np.array(f['images']['labels'])
    #img_mean = np.array(f['images']['data_mean'])
    #matlab store data column wise so we need to transpose it
    return data.transpose(), labels#, img_mean.transpose()
    
  def read_mean(self,mat_file):
    """
    mat file variables:
    data_mean       #
    """
    assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
    f = h5.File(mat_file)
    data_mean = np.array(f['data_mean'])
    return data_mean.transpose()

class matlab(object):
  """designed to replace mat2py
  manipulate .mat file given the list of the variables
  """
  def __init__(self, mat_file):
    self.f = h5.File(mat_file)
  def load(self, var):
    pass

###############################################################################
class sbir(object):
  """class for image retrieval
  """
  def __init__(self,params):
    if 'data' in params:
      self.data = np.load(params['data'])
    if 'queries' in params:
      self.queries = np.load(params['queries'])
    #self.pool = ThreadPool()
  
  def load_database(self, data, labels=None):
    if isinstance(data,basestring): #path to database
      assert data.endswith('.npz'), 'Database must be a .npz file'
      self.data = np.load(data)
    elif isinstance(data,np.ndarray): #numpy array NxD
      self.data = {}
      self.data['feats'] = data
      if labels is not None:
        assert data.shape[0] == labels.size, 'Error: dimension mismatch bw data and label'
        self.data['labels'] = labels
    else:
      print('Error: data is neither .npz database or numpy array. Exit now.')
      sys.exit()
  
  def dist_L2(self, a_query):
    """
    Eucludean distance between a (single) query & all features in database
    used in pdist
    """
    return np.sum((a_query-self.data['feats'])**2,axis=1)
    
  def pdist(self, query_feat):
    """
    Compute distance (L2) between queries and data features
    query_feat: NxD numpy array with N features of D dimensions
    """
    if len(query_feat.shape) <2:   #single query
      query_feat = query_feat[None,...]
    res = [np.sum((query_feat[i]-self.data['feats'])**2,axis=1) for i in range(query_feat.shape[0])]
#==============================================================================
#     threads = [self.pool.apply_async(self.dist_L2,(query_feat[i],)) for \
#                 i in range(query_feat.shape[0])]
#     res = [thread.get() for thread in threads]
#==============================================================================
    res = np.array(res)
    return res
  
  def retrieve(self,queries):
    """retrieve: return the indices of the data points in relevant order
    """
    if isinstance(queries,basestring):
      query_feat = np.load(queries)['feats']
    elif isinstance(queries, np.ndarray):
      query_feat = queries
    else:
      print('Error: queries must be a .npz file or ndarray')
      sys.exit()
    
    res = self.pdist(query_feat)
    return res.argsort()
  
  def retrieve2file(self,queries, out_file, num=0):
    """perform retrieve but write output to a file
    num specifies number of returned images, if num==0, all images are returned
    """
    res = self.retrieve(queries)
    if num > 0 and num < self.data['feats'].shape[0]:
      res = res[:,0:num]
    if out_file.endswith('.npz'):
      np.savez(out_file, results = res)
    return 0
  
  def retrieve2file_hardsel(self,queries, out_file, num=0, qsf = 2.0):
    """same as retrieve2file but highly customised for hard triplet selection
    Hard negative: exclude images of the same cat in the retrieving process
    Hard positive: include only images of the same cat and choose the farthest
    queries: must be a .npz file containing feats & labels, e.g. output of extract_cnn_feat()
    num: number of returned results
    """
    queries_ = np.load(queries)
    query_feat = queries_['feats'] * qsf
#==============================================================================
#     """old code"""
#     query_label = queries_['labels']
#     # compute L2 distance
#     dist = [np.sum((query_feat[i]-self.data['feats'])**2,axis=1) for i in range(query_feat.shape[0])]
#     # returned image id
#     ids = [dist[i].argsort() for i in range(len(dist))]
#     #labels of the returned images
#     ret_labels = [self.data['labels'][ids[i]] for i in range(len(ids))]
#     # relevant
#     rel = [ret_labels[i] == query_label[i] for i in range(len(ret_labels))]
#     #include/exclude the relevant
#     pos = [ids[i][rel[i]] for i in range(len(ids))]
#     pos = np.fliplr(np.array(pos))   #hard positive
#     neg = [ids[i][~rel[i]] for i in range(len(ids))]
#     neg = np.array(neg)              #hard negative
#==============================================================================
    """new code"""
    query_label = queries_['labels'][...,None]
    ids = self.retrieve(query_feat)
    ret_labels = self.data['labels'][ids]
    # relevant
    rel = ret_labels == query_label
    #include/exclude the relevant in hard pos/neg selection
    pos = ids[rel].reshape([rel.shape[0],-1])
    pos = np.fliplr(pos)                       #hard positive
    neg = ids[~rel].reshape([rel.shape[0],-1]) #hard negative
    
    if num > 0 and num < self.data['feats'].shape[0]:
      pos = pos[:,0:num]
      neg = neg[:,0:num]
    if out_file.endswith('.npz'):
      np.savez(out_file, pos = pos, neg = neg)
    
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP
    
  def compute_mAP(self,queries,qsf = 1.0):
    """
    compute mAP given queries in .npz format
    """
    assert isinstance(queries,basestring) and queries.endswith('.npz'),'Opps! Input must be a .npz file'
    tmp = np.load(queries)
    query_feat = tmp['feats'] * qsf
    query_label = tmp['labels'][...,None]
    ids = self.retrieve(query_feat)
    ret_labels = self.data['labels'][ids]
    # relevant
    rel = ret_labels == query_label
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP

class sbir_hardsel(object):
  """sbir for hard selection
  similar to sbir, __init is a bit different
  """
  def __init__(self,params, qsf = 1.0):
    self.data_src = self.data_q = self.label_src = self.label_q = None
    self.qsf = qsf
    if 'data_src' in params:
      self.data_src = params['data_src']
    if 'data_q' in params:
      self.data_q = params['data_q']
      if len(self.data_q.shape) < 2: #single query
        self.data_q = self.data_q[None,...]
    if 'label_src' in params:
      self.label_src = np.array(params['label_src'])
    if 'label_q' in params:
      self.label_q = np.array(params['label_q'])
      if len(self.label_q.shape) < 2:
        self.label_q = self.label_q[...,None]
  
  def update(self,params):
    if 'data_src' in params:
      self.data_src = params['data_src']
    if 'data_q' in params:
      self.data_q = params['data_q']
      if len(self.data_q.shape) < 2: #single query
        self.data_q = self.data_q[None,...]
    if 'label_src' in params:
      self.label_src = np.array(params['label_src'])
    if 'label_q' in params:
      self.label_q = np.array(params['label_q'])
      if len(self.label_q.shape) < 2:
        self.label_q = self.label_q[...,None]
        
  def pdist(self):
    """
    Compute distance (L2) between queries and data features
    query_feat: NxD numpy array with N features of D dimensions
    """
    
    res = [np.sum((self.data_q[i]*self.qsf-self.data_src)**2,axis=1) for i in range(self.data_q.shape[0])]
    res = np.array(res)
    return res
  
  def retrieve(self,save = '',num=5):
    """retrieve: return the indices of the data points in relevant order
    """
    res = self.pdist().argsort()
    if save and save.endswith('.npz'):
      np.savez(save,res = res[:,:num])
    return res
  
  def computeMAP(self):
    """compute retrieve mAP"""
    ids = self.retrieve()
    ret_labels = self.label_src[ids]
    rel = ret_labels == self.label_q
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP
  
  def retrieve2file(self, out_file, numn=0, nump=0):
    """highly customised for hardsel"""
    ids = self.retrieve()
    ret_labels = self.label_src[ids]
    rel = ret_labels == self.label_q
    #include/exclude the relevant in hard pos/neg selection
    pos = ids[rel].reshape([rel.shape[0],-1])
    pos = np.fliplr(pos)                       #hard positive
    neg = ids[~rel].reshape([rel.shape[0],-1]) #hard negative
    
    if nump > 0 and nump < pos.shape[1]:
      pos = pos[:,0:nump]
    if numn > 0 and numn < neg.shape[1]:
      neg = neg[:,0:numn]
    if out_file.endswith('.npz'):
      np.savez(out_file, pos = pos, neg = neg)
    
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP
    
class decay(object):
  def __init__(self,params):
    if 'k' in params:
      self.k = params['k']
    if 'max_iter' in params:
      self.max_iter = params['max_iter']
    if 'bias' in params:
      self.bias = params['bias']
    if 'power' in params:
      self.power = params['power']
  def poly_decay(self,x):
    return int(self.k * (1 - float(x)/self.max_iter)**self.power + self.bias)
    
    