import xml.etree.ElementTree as ET
import cairo
import rsvg
import random
import copy
import numpy as np


class SVGProcessor():
    """
    class responsible for manipulating and processing svg files.
    it can occlude strokes and convert resulting svgs to numpy arrays
    """

    def __init__(self, filename):
        """
        read svg file from disk and get xml root
        """
        # open .svg as a xml file
        self.tree_org = ET.parse(filename)
        self.tree_root = None

    def randomly_occlude_strokes(self, n=4):
      """
      given a number n, divide number of strokes by n and create groups of
      n strokes. Randomly remove stroke groups, effectively occluding some
      related-by-order strokes from the original sketch
      """
      self.tree = copy.deepcopy(self.tree_org)
      self.tree_root = self.tree.getroot()

      # manipulate the xml components
      # get root path
      gg = self.tree_root.find(
          "./{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}g"
      )
      # get all paths in this .svg
      paths = self.tree_root.findall(".//{http://www.w3.org/2000/svg}path")
      
        # split paths in n groups
      sid = np.linspace(0,len(paths),n+1,dtype = np.uint8)
      if len(paths) < n:
        sid = (n+1) * [len(paths)]
      
      #  remove all paths
      [gg.remove(p) for p in paths]
      
      # randomly select number of accumulated strokes
      n_sel_acc_strokes = random.randint(1, n)
      
      # accumulate stroke groups up until random stopping
      for path in paths[:sid[n_sel_acc_strokes]]:
        gg.append(path)

#==============================================================================
#         n_acc_strokes = len(paths)/n
#         if n_acc_strokes == 0:
#             n_acc_strokes = 1
# 
#         # possibly the most pythonic thing I have ever written,
#         # basically, create an array of strokes with every n_acc_strokes
#         accumulated_paths = [
#             paths[i:i+n_acc_strokes]
#             for i in range(0, len(paths), n_acc_strokes)
#             if i+n_acc_strokes < len(paths)
#         ]
# 
#         #  and remove all paths
#         [gg.remove(p) for p in paths]
# 
#         # randomly select number of accumulated strokes
#         n_sel_acc_strokes = random.randint(1, n)
# 
#         # accumulate stroke groups up until random stopping
#         for index, acc_path in enumerate(
#                     accumulated_paths[:n_sel_acc_strokes]):
#             for path in acc_path:
#                 gg.append(path)
#==============================================================================

    def get_numpy_array(self, size=256, scale=0.333):
        """
        using rsvg library, render the svg into a cairo surface with memory
        defined by a numpy array, effectively rendering the svg to an array
        """
        if self.tree_root is None:
          self.tree_root = self.tree_org.getroot()
        
        # get the result into a svg object (using rsvg lib)
        svg_object = rsvg.Handle(ET.tostring(self.tree_root))

        # create a numpy array to use as our canvas
        data = np.zeros((size, size, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            data, cairo.FORMAT_ARGB32, size, size)

        cr = cairo.Context(surface)

        # fill with solid white and set scale (magic scale number)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()
        cr.scale(scale, scale)

        # render our manipulated svg into cairo surface
        svg_object.render_cairo(cr)

        return data
    def get_num_strokes(self):
      """
      return number of strokes in this svg
      """
      tree_root = self.tree_org.getroot()
      paths = tree_root.findall(".//{http://www.w3.org/2000/svg}path")
      return len(paths)
    
    def remove_strokes_and_convert(self,n=4,size=256, scale = 0.333):
      """
      combine the 2 above functions: randomly_occlude_stroke() and 
      get_numpy_array()
      Hopefully it will make this class thread safe i.e. an object instance 
      can be called simultanneously
      modified to work with multi-thread in triplet_data layer
      """
      
      """The randomly_occlude_strokes part"""
      tree = copy.deepcopy(self.tree_org)
      tree_root = tree.getroot()

      # manipulate the xml components
      # get root path
      gg = tree_root.find(
          "./{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}g"
      )
      # get all paths in this .svg
      paths = tree_root.findall(".//{http://www.w3.org/2000/svg}path")
      
      ##keep 50% first strokes, for the rest randomly discard some of them
      num_str = len(paths)
      if num_str > 10:
        nkeeps = max(10,num_str/2)
        rest = range(nkeeps,num_str)  #the rest is potentially discarded
        random.shuffle(rest)
        ndiscard = random.randrange(0,len(rest)) #number of strokes to be discarded
        discards = rest[:ndiscard]              #id of strokes to be discarded
        
        [gg.remove(paths[p]) for p in discards]
        
#==============================================================================
#       ##group to 25%, 50%, 75% and 100%
#       # split paths in n groups
#       sid = np.linspace(0,len(paths),n+1,dtype = np.uint8)
#       if len(paths) < n:
#         sid = (n+1) * [len(paths)]
#       
#       #  remove all paths
#       [gg.remove(p) for p in paths]
#       
#       # randomly select number of accumulated strokes
#       n_sel_acc_strokes = random.randint(1, n)
#       
#       # accumulate stroke groups up until random stopping
#       for path in paths[:sid[n_sel_acc_strokes]]:
#         gg.append(path)
#==============================================================================
      
      """the get_numpy_array part"""
      # get the result into a svg object (using rsvg lib)
      svg_object = rsvg.Handle(ET.tostring(tree_root))

      # create a numpy array to use as our canvas
      data = np.zeros((size, size, 4), dtype=np.uint8)
      surface = cairo.ImageSurface.create_for_data(
          data, cairo.FORMAT_ARGB32, size, size)

      cr = cairo.Context(surface)

      # fill with solid white and set scale (magic scale number)
      cr.set_source_rgb(1.0, 1.0, 1.0)
      cr.paint()
      cr.scale(scale, scale)

      # render our manipulated svg into cairo surface
      svg_object.render_cairo(cr)

      return data
