'''
Create raw data pickle file
'''
import numpy as np
import pickle
import re
import os
from PIL import Image
from collections import defaultdict

# Script config
RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
GRAYSCALE = False  # convert image to grayscale
TARGET_W, TARGET_H = 640, 480  # 1.74 is weighted avg ratio, but 1.65 aspect ratio is close enough (1.65 was for stop signs)

###########################
# Execute main script
###########################

# First get mapping from sign name string to integer label
sign_map = {'danger': 1, 'prohibitory': 2, 'mandatory':3}  # only 3 sign classes (background class is 0)

# Create raw data pickle file
'''
data_raw{
 source1: {image_filename : [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...],image_filename2:...}
 source2: {image_filename : [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]}
 source3: {image_filename : [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]}
 source4: {image_filename : [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]}
}
'''
data_raw = defaultdict(dict)
path='/home/simran/Desktop/dataset'
os.chdir(path)
folder_list = os.listdir(path)

for folder in folder_list:
	# For speed, put entire contents of mergedAnnotations.csv in memory
	merged_annotations = []
	with open(folder +'/Annotations.csv', 'r') as f: #open folder_name.csv
		for line in f:
			line = line[:-1]  # strip trailing newline
			merged_annotations.append(line)

	# Create pickle file to represent dataset

	image_files = os.listdir(folder+'/images') 
	for image_file in image_files:
		# Find box coordinates for all signs in this image
		class_list = []
		box_coords_list = []
		for line in merged_annotations: 
			if re.search(image_file, line):
				fields = line.split(',')
				# Get sign name and assign class label
				sign_name = fields[-1]
				if sign_name != 'warning' and sign_name != 'prohibitory' and sign_name != 'mandatory':
					continue  # ignore signs that are neither stop nor pedestrianCrossing signs
				sign_class = sign_map[sign_name]
				class_list.append(sign_class)

				# Resize image, get rescaled box coordinates
				box_coords = np.array([float(x) for x in fields[1:5]])
				image = Image.open(folder+"/images/"+image_file)
				
				if GRAYSCALE:
					image = image.convert('L')  # 8-bit grayscale
				if RESIZE_IMAGE:
					orig_w, orig_h = image.size
					image = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)  # high-quality downsampling filter

					resized_dir = 'resized_images_%dx%d/' % (TARGET_W, TARGET_H)
					if not os.path.exists(resized_dir):
						os.makedirs(resized_dir)

					image.save(os.path.join(resized_dir, image_file))

					# Rescale box coordinates
					x_scale = TARGET_W / orig_w
					y_scale = TARGET_H / orig_h

					ulc_x, ulc_y, lrc_x, lrc_y = box_coords
					new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
					new_box_coords = [round(x) for x in new_box_coords]
					box_coords = np.array(new_box_coords)

				box_coords_list.append(box_coords)

		if len(class_list) == 0:
			continue  # ignore images with no signs-of-interest
		class_list = np.array(class_list)
		box_coords_list = np.array(box_coords_list)

		# Create the list of dicts
		the_list = []
		for i in range(len(box_coords_list)):
			d = {'class': class_list[i], 'box_coords': box_coords_list[i]}
			the_list.append(d)
		data_raw[folder][image_file] = the_list
	print(data_raw)

with open('/home/simran/Desktop/data_raw_%dx%d.p' % (TARGET_W, TARGET_H), 'wb') as f:
	pickle.dump(data_raw, f)
                    
