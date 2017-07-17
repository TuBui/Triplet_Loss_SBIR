#! /usr/bin/python
# script to break .svg files into multiple accumulated stroke as numpy arrays
# usage: ./manipulate-svg filename.svg

import sys
import matplotlib.pyplot as plt
import SVGProcessor

# check input
if len(sys.argv) < 2:
    print("usage: ./manipulate-svg filename.svg")
    sys.exit(2)

for arg in sys.argv:
    if arg == "-h":
        print("usage: ./manipulate-svg filename.svg")
        sys.exit(0)

filename = str(sys.argv[1])

processor = SVGProcessor(filename)
processor.randomly_occlude_strokes()
data = processor.get_numpy_array()

plt.imshow(data)
plt.show()
