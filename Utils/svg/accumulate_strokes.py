#! /usr/bin/python
# script to break .svg files into multiple accumulated stroke files
# usage: ./accumulate_strokes n [--divide] [--accumulate]

import sys
import os
import xml.etree.ElementTree as ET

division = 1
uses_division = False
accumulate_strokes = False
n_acc_strokes = 1

# check input
if len(sys.argv) < 2:
    print("usage: ./accumulate_strokes n")
    sys.exit(2)

for arg in sys.argv:
    if arg == "-h":
        print("usage: ./accumulate_strokes n")
        sys.exit(0)
    elif arg == "--divide":
        division = int(sys.argv[1])
        uses_division = True
    elif arg == "--accumulate":
        division = int(sys.argv[1])
        uses_division = True
        accumulate_strokes = True

n_acc_strokes = int(sys.argv[1])

progress_counter = 0

# open dataset folder and look for every folder inside
data_folders = [
    os.path.join(os.getcwd() + '/dataset', d)
    for d in sorted(os.listdir(os.getcwd() + '/dataset'))
    if os.path.isdir(os.path.join(os.getcwd() + '/dataset', d))
]

for data_folder in data_folders:
    data_files = [
        f for f in sorted(os.listdir(data_folder))
        if not os.path.isdir(
            os.path.join(data_folder, f)
        ) and ".DS_Store" not in f
    ]

    for image_file in data_files:

        # file name without extension and full path name
        full_path = data_folder + '/' + image_file
        file_name_w_ext = os.path.splitext(os.path.splitext(full_path)[0])[0]

        # create a directory with the image's name
        if not os.path.exists(file_name_w_ext):
            os.makedirs(file_name_w_ext)

        # open .svg as a xml file
        tree = ET.parse(full_path)
        tree_root = tree.getroot()

        # get all paths in this .svg
        paths = tree_root.findall(".//{http://www.w3.org/2000/svg}path")

        if uses_division:
            n_acc_strokes = len(paths)/division
            if n_acc_strokes == 0:
                n_acc_strokes = 1

        # possibly the most pythonic thing I have ever written, 
        # basically, create an array of strokes with every n_acc_strokes
        accumulated_paths = [
            paths[i:i+n_acc_strokes]
            for i in range(0, len(paths), n_acc_strokes)
            if i+n_acc_strokes < len(paths)
        ]

        # if not accumulating save a full sketch
        if not accumulate_strokes:
            tree.write(os.path.join(file_name_w_ext, '0.svg'))
        
        # get root path
        gg = tree_root.find(
            "./{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}g"
        )

        #  and remove all paths
        [gg.remove(p) for p in paths]

        # for each accumulated path, create new version of file
        for index, acc_path in enumerate(accumulated_paths):

            # if we are using division, make pairwise combinations as in Sketch-a-net
            if index != 0 and uses_division and not accumulate_strokes:
                for path in acc_path:
                    gg.append(path)
                tree.write(os.path.join(file_name_w_ext, str(index)+"_"+str(index+1)+".svg"))

            # if not accumulating, clean the slate
            if not accumulate_strokes:
                [gg.remove(p) for p in paths if p in gg]
            for path in acc_path:
                gg.append(path)

            # and write new svg
            tree.write(os.path.join(file_name_w_ext, str(index+1)+ ".svg"))

        # print progress every once in a while
        if ((progress_counter % 1000) == 0): print (full_path + '...done!')
        progress_counter += 1