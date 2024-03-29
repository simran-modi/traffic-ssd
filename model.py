'''
Model definition
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from settings import *
from data_prep import calc_iou


def SSDHook(feature_map, hook_id):
	"""
	Takes input feature map, output the predictions tensor
	hook_id is for variable_scope unqie string ID
	"""
	with tf.variable_scope('ssd_hook_' + hook_id):
		# Note we have linear activation (i.e. no activation function)
		net_conf = slim.conv2d(inputs=feature_map, num_outputs=NUM_PRED_CONF, kernel_size=[3, 3], activation_fn=None, scope='conv_conf')
		net_conf = tf.contrib.layers.flatten(net_conf)
		net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [3, 3], activation_fn=None, scope='conv_loc')
		net_loc = tf.contrib.layers.flatten(net_loc)
	return net_conf, net_loc


def ModelHelper(y_pred_conf, y_pred_loc):
	"""
	Define loss function, optimizer, predictions, and accuracy metric
	Loss includes confidence loss and localization loss

	conf_loss_mask is created at batch generation time, to mask the confidence losses
	It has 1 at locations w/ positives, and 1 at select negative locations)
	such that negative-to-positive ratio of NEG_POS_RATIO is satisfied

	Arguments:
		* y_pred_conf: Class predictions from model,
			a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * num_classes]
		* y_pred_loc: Localization predictions from model,
			a tensor of shape [batch_size, num_feature_map_cel)ls * num_defaul_boxes * 4]

	Returns relevant tensor references
	"""
	num_total_preds = 0
	for fm_size in FM_SIZES:
		num_fm_cells = fm_size[0] * fm_size[1]
		num_total_preds +=  num_fm_cells * NUM_DEFAULT_BOXES
	num_total_preds_conf = num_total_preds * NUM_CLASSES
	num_total_preds_loc  = num_total_preds * 4

	# Input tensors
	y_true_conf = tf.placeholder(tf.int32, [None, num_total_preds], name='y_true_conf')  # classification ground-truth labels
	y_true_loc  = tf.placeholder(tf.float32, [None, num_total_preds_loc], name='y_true_loc')  # localization ground-truth labels
	conf_loss_mask = tf.placeholder(tf.float32, [None, num_total_preds], name='conf_loss_mask')  # 1 mask "bit" per def. box

	num_pos = tf.placeholder(tf.float32, name='num_pos')	#Number of matching default boxes
	# Confidence loss
	logits = tf.reshape(y_pred_conf, [-1, num_total_preds, NUM_CLASSES])
	conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_conf, logits=logits)
	conf_loss = conf_loss_mask * conf_loss  # "zero-out" the loss for don't-care negatives
	conf_loss = tf.reduce_sum(conf_loss)

	# Localization loss (smooth L1 loss)
	# loc_loss_mask is analagous to conf_loss_mask, except 4 times the size
	diff = y_true_loc - y_pred_loc 
	
	loc_loss_l2 = 0.5 * (diff**2.0)
	loc_loss_l1 = tf.abs(diff) - 0.5
	smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
	loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)
	#tf.Print(loc_loss,[loc_loss],"Localization loss: ")
	loc_loss_mask = tf.minimum(y_true_conf, 1)  # have non-zero localization loss only where we have matching ground-truth box
	loc_loss_mask = tf.to_float(loc_loss_mask)
	loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)  # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
	loc_loss_mask = tf.reshape(loc_loss_mask, [-1, num_total_preds_loc])  # removing the inner-most dimension of above
	loc_loss = loc_loss_mask * loc_loss
	loc_loss = tf.reduce_sum(loc_loss)

	# Weighted average of confidence loss and localization loss-zero localization loss only where we ha
	# Also add regularization loss
	loss = ((conf_loss + LOC_LOSS_WEIGHT * loc_loss)/num_pos) + tf.reduce_sum(slim.losses.get_regularization_losses())
	optimizer = OPT.minimize(loss)

	#reported_loss = loss #tf.reduce_sum(loss, 1)  # DEBUG

	# Class probabilities and predictions
	probs_all = tf.nn.softmax(logits)
	probs, preds_conf = tf.nn.top_k(probs_all)  # take top-1 probability, and the index is the predicted class
	probs = tf.reshape(probs, [-1, num_total_preds])
	preds_conf = tf.reshape(preds_conf, [-1, num_total_preds])

	# Return a dictionary of {tensor_name: tensor_reference}
	ret_dict = {
		'y_true_conf': y_true_conf,
		'y_true_loc': y_true_loc,
		'conf_loss_mask': conf_loss_mask,
		'optimizer': optimizer,
		'conf_loss': conf_loss,
		'loc_loss': loc_loss,
		'loss': loss,
		'probs': probs,
		'preds_conf': preds_conf,
		'preds_loc': y_pred_loc,
		'num_pos' : num_pos,
	}
	return ret_dict


def AlexNet():
	"""
	AlexNet
	"""
	# Image batch tensor and dropout keep prob placeholders
	x = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, NUM_CHANNELS], name='x')
	is_training = tf.placeholder(tf.bool, name='is_training')

	# Classification and localization predictions
	preds_conf = []  # conf -> classification b/c confidence loss -> classification loss
	preds_loc = []

	# Create the variables and copy the weights
	wt = np.load("/home/simran/Desktop/project/weights.npy", encoding = 'latin1')
	wt_dict = wt.item()
	#print(wt_dict['Convolution1']['weights'])
	conv1_weights = tf.get_variable('conv1_weights', initializer=wt_dict['Convolution1']['weights'], trainable=True)
	conv1_biases = tf.get_variable('conv1_biases', initializer=wt_dict['Convolution1']['biases'], trainable=True)
	conv2_weights = tf.get_variable('conv2_weights', initializer=wt_dict['Convolution2']['weights'], trainable=True)
	conv2_biases = tf.get_variable('conv2_biases', initializer=wt_dict['Convolution2']['biases'], trainable=True)
	conv3_weights = tf.get_variable('conv3_weights', initializer=wt_dict['Convolution3']['weights'], trainable=True)
	conv3_biases = tf.get_variable('conv3_biases', initializer=wt_dict['Convolution3']['biases'], trainable=True)
	conv4_weights = tf.get_variable('conv4_weights', initializer=wt_dict['Convolution4']['weights'], trainable=True)
	conv4_biases = tf.get_variable('conv4_biases', initializer=wt_dict['Convolution4']['biases'], trainable=True)
	conv5_weights = tf.get_variable('conv5_weights', initializer=wt_dict['Convolution5']['weights'], trainable=True)
	conv5_biases = tf.get_variable('conv5_biases', initializer=wt_dict['Convolution5']['biases'], trainable=True)
	conv6_weights = tf.get_variable('conv6_weights', initializer=wt_dict['Convolution6']['weights'], trainable=True)
	conv6_biases = tf.get_variable('conv6_biases', initializer=wt_dict['Convolution6']['biases'], trainable=True)

	#Build the network
	#layer 1
	convolution_1 = tf.nn.conv2d(x, conv1_weights, strides=[1,4,4,1], padding='VALID', name='conv1') #no padding
	bias_layer_1 = tf.nn.relu(tf.nn.bias_add(convolution_1, conv1_biases)) #bias shape is same as last dimension of input
	normalized_layer_1 = tf.nn.local_response_normalization(bias_layer_1, depth_radius=5,bias=2,alpha=0.0005,beta=0.75,name ='norm1')
	pooled_layer_1 = tf.nn.max_pool(normalized_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

	#layer 2
	convolution_2 = tf.nn.conv2d(pooled_layer_1, conv2_weights, strides=[1,1,1,1], padding='SAME', name='conv2') #pads zero on either side of input
	bias_layer_2 = tf.nn.relu(tf.nn.bias_add(convolution_2, conv2_biases))
	normalized_layer_2 = tf.nn.local_response_normalization(bias_layer_2, depth_radius=5,bias=8,alpha=0.0005,beta=0.75,name ='norm2')
	pooled_layer_2 = tf.nn.max_pool(normalized_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

	net_conf, net_loc = SSDHook(convolution_2, 'conv2')
	preds_conf.append(net_conf)
	preds_loc.append(net_loc)

	#layer 3
	convolution_3 = tf.nn.conv2d(pooled_layer_2, conv3_weights, strides=[1,1,1,1], padding='SAME', name='conv3')
	bias_layer_3 = tf.nn.relu(tf.nn.bias_add(convolution_3, conv3_biases))

	#layer 4
	convolution_4 = tf.nn.conv2d(bias_layer_3, conv4_weights, strides=[1,1,1,1], padding='SAME', name='conv4') 
	bias_layer_4 = tf.nn.relu(tf.nn.bias_add(convolution_4, conv4_biases))
	
	#layer 5
	convolution_5 = tf.nn.conv2d(bias_layer_4, conv5_weights, strides=[1,1,1,1], padding='SAME', name='conv5') 
	bias_layer_5 = tf.nn.relu(tf.nn.bias_add(convolution_5, conv5_biases))
	pooled_layer_5 = tf.nn.max_pool(bias_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

	net_conf, net_loc = SSDHook(convolution_5, 'conv5')
	preds_conf.append(net_conf)
	preds_loc.append(net_loc)

	#layer 6
	convolution_6 = tf.nn.conv2d(pooled_layer_5, conv6_weights, strides=[1,1,1,1], padding='SAME', name='conv6') 
	bias_layer_6 = tf.nn.relu(tf.nn.bias_add(convolution_6, conv6_biases))
	dropout_layer_6 = tf.nn.dropout(bias_layer_6, keep_prob = 0.5, name='drop6')

	net_conf, net_loc = SSDHook(convolution_6, 'conv6')
	preds_conf.append(net_conf)
	preds_loc.append(net_loc)

	# Concatenate all preds together into 1 vector, for both classification and localization predictions
	final_pred_conf = tf.concat(preds_conf,1)
	final_pred_loc = tf.concat(preds_loc,1)

	# Return a dictionary of {tensor_name: tensor_reference}
	ret_dict = {
		'x': x,
		'y_pred_conf': final_pred_conf,
		'y_pred_loc': final_pred_loc,
		'is_training': is_training,
	}
	return ret_dict


def SSDModel():
	"""
	Wrapper around the model and model helper
	Returns dict of relevant tensor references
	"""
	if MODEL == 'AlexNet':
		model = AlexNet()
	else:
		raise NotImplementedError('Model %s not supported' % MODEL)
	model_helper = ModelHelper(model['y_pred_conf'], model['y_pred_loc'])

	ssd_model = {}
	for k in model.keys():
		ssd_model[k] = model[k]
	for k in model_helper.keys():
		ssd_model[k] = model_helper[k]

	return ssd_model


def nms(y_pred_conf, y_pred_loc, prob):
	"""
	Non-Maximum Suppression (NMS)
	Performs NMS on all boxes of each class where predicted probability > CONF_THRES
	For all boxes exceeding IOU threshold, select the box with highest confidence
	Returns a lsit of box coordinates post-NMS

	Arguments:
		* y_pred_conf: Class predictions, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)
		* y_pred_loc: Bounding box coordinates, numpy array of shape (num_feature_map_cells * num_defaul_boxes * 4,)
			These coordinates are normalized coordinates relative to center of feature map cell
		* prob: Class probabilities, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)

	Returns:
		* boxes: Numpy array of boxes, with shape (num_boxes, 6). shape[0] is interpreted as:
			[x1, y1, x2, y2, class, probability], where x1/y1/x2/y2 are the coordinates of the
			upper-left and lower-right corners. Box coordinates assume the image size is IMG_W x IMG_H.
			Remember to rescale box coordinates if your target image has different dimensions.
	"""
	# Keep track of boxes for each class
	class_boxes = {}  # class -> [(x1, y1, x2, y2, prob), (...), ...]
	with open('signnames.csv', 'r') as f:
		for line in f:
			cls, _ = line.split(',')
			class_boxes[float(cls)] = []

	# Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
	y_idx = 0
	for fm_size in FM_SIZES:
		fm_h, fm_w = fm_size  # feature map height and width
		for row in range(fm_h):
			for col in range(fm_w):
				for db in DEFAULT_BOXES:
					# Only perform calculations if class confidence > CONF_THRESH and not background class
					if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
						# Calculate absolute coordinates of predicted bounding box
						xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
						center_coords = np.array([xc, yc, xc, yc])
						abs_box_coords = center_coords + y_pred_loc[y_idx*4 : y_idx*4 + 4]  # predictions are offsets to center of fm cell

						# Calculate predicted box coordinates in actual image
						scale = np.array([IMG_W/fm_w, IMG_H/fm_h, IMG_W/fm_w, IMG_H/fm_h])
						box_coords = abs_box_coords * scale
						box_coords = [int(round(x)) for x in box_coords]

						# Compare this box to all previous boxes of this class
						cls = y_pred_conf[y_idx]
						cls_prob = prob[y_idx]
						#box = (*box_coords, cls, cls_prob) #python3.5 + FIXME
						box = (box_coords, cls, cls_prob)
						if len(class_boxes[cls]) == 0:
							class_boxes[cls].append(box)
						else:
							suppressed = False  # did this box suppress other box(es)?
							overlapped = False  # did this box overlap with other box(es)?
							for other_box in class_boxes[cls]:
								iou = calc_iou(box[:4], other_box[:4])
								if iou > NMS_IOU_THRESH:
									overlapped = True
									# If current box has higher confidence than other box
									if box[5] > other_box[5]:
										class_boxes[cls].remove(other_box)
										suppressed = True
							if suppressed or not overlapped:
								class_boxes[cls].append(box)

					y_idx += 1

	# Gather all the pruned boxes and return them
	boxes = []
	for cls in class_boxes.keys():
		for class_box in class_boxes[cls]:
			boxes.append(class_box)
	boxes = np.array(boxes)

	return boxes
