import numpy as np
import tensorflow as tf 

#unzip weights.npy.zip file first
weight_path = "~/Desktop/project/scripts/weights.npy"

#loads the pre-trained network weights imported from caffe
def load_pre_trained_weights(path):
	wt = np.load("weights.npy", encoding = 'latin1') #return 0-d numpy array containing a dictionary
	wt_dict = wt.item()
	return wt_dict


wt_dict = load_pre_trained_weights(weight_path)

# Create the variables and copy the weights
conv1_weights = tf.get_variable('conv1_weights', initializer=wt_dict['Convolution1']['weights'], trainable=False)
conv1_biases = tf.get_variable('conv1_biases', initializer=wt_dict['Convolution1']['biases'], trainable=False)
conv2_weights = tf.get_variable('conv2_weights', initializer=wt_dict['Convolution2']['weights'], trainable=False)
conv2_biases = tf.get_variable('conv2_biases', initializer=wt_dict['Convolution2']['biases'], trainable=False)
conv3_weights = tf.get_variable('conv3_weights', initializer=wt_dict['Convolution3']['weights'], trainable=False)
conv3_biases = tf.get_variable('conv3_biases', initializer=wt_dict['Convolution3']['biases'], trainable=False)
conv4_weights = tf.get_variable('conv4_weights', initializer=wt_dict['Convolution4']['weights'], trainable=False)
conv4_biases = tf.get_variable('conv4_biases', initializer=wt_dict['Convolution4']['biases'], trainable=False)
conv5_weights = tf.get_variable('conv5_weights', initializer=wt_dict['Convolution5']['weights'], trainable=False)
conv5_biases = tf.get_variable('conv5_biases', initializer=wt_dict['Convolution5']['biases'], trainable=False)
conv6_weights = tf.get_variable('conv6_weights', initializer=wt_dict['Convolution6']['weights'], trainable=False)
conv6_biases = tf.get_variable('conv6_biases', initializer=wt_dict['Convolution6']['biases'], trainable=False)

print(conv1_weights.get_shape())
print(conv2_weights.get_shape())
print(conv3_weights.get_shape())
print(conv4_weights.get_shape())
print(conv5_weights.get_shape())
print(conv6_weights.get_shape())

#Create the base network -Tshingua Tencent
x = tf.get_variable("input_image", [1,480, 640,3]) #batch, in_height, in_width, in_channels
#default data format - batch, height, width, channels (NHWC) for strides
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


#layer 6
convolution_6 = tf.nn.conv2d(pooled_layer_5, conv6_weights, strides=[1,1,1,1], padding='SAME', name='conv6') 
bias_layer_6 = tf.nn.relu(tf.nn.bias_add(convolution_6, conv6_biases))
dropout_layer_6 = tf.nn.dropout(bias_layer_6, keep_prob = 0.5, name='drop6')



