#Convolutional neural network (CNN) for SERS spectra. 
#
#the present CNN was applied with Dynamic-SERS optophysiology for metabolites classification near cells.
#
# Felix Lussier, Université de Montréal, July 2018.



# Import modules and methods. 
import tensorflow as tf
import matplotlib.pyplot as ptl
%matplotlib inline
import itertools
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import interp
import math


# loading of the training data:

# Data were preprocessed in MatLab and labelled.
# YOUR_DATA must have the size npixels x number of spectra. 
# The first row must be the index number of the spectrum, starting from 0.
# YOUR_LABEL must have the size of number of spectrum x number of labels.
#The fist row must be the index number of labels, starting from 0. 
#A value of 1 must be place to the appropriate column for the corresponding labels


	Data = pd.read_csv('YOUR_DATA') #data generated from MatLab
	encode_labels = pd.read_csv('YOUR_LABEL') #data generated from MatLab
	df = pd.DataFrame(data = Data)
	labels = pd.DataFrame(data=encode_labels)

# Scaling data form 0 to 1
	scaler_model = MinMaxScaler()
	scaler_model.fit(df)
	scale_data = scaler_model.transform(df)
	Data = pd.DataFrame(data=scale_data)

# Train, test, validation data splitting
	X_train, x_test, y_train, y_test = train_test_split(Data.transpose(),labels,test_size=0.2)
	x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
	
	class_names = ['NAME_OF_YOUR_LABELS'] #Update according to the number of analyte probed.

# Batch normalization code:

	def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
		exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) 
		
		# adding the iteration prevents from averaging across non-existing iterations
		
		bnepsilon = 1e-5
		if convolutional:
			mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
		else:
			mean, variance = tf.nn.moments(Ylogits, [0])
		update_moving_averages = exp_moving_avg.apply([mean, variance])
		m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
		v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
		Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
		return Ybn, update_moving_averages

	def compatible_convolutional_noise_shape(Y):
		noiseshape = tf.shape(Y)
		noiseshape = noiseshape * tf.constant([1,0,1]) + tf.constant([0,1,0])
		return noiseshape

# Batch feeder

	class Opto():
		
		def __init__(self):
			self.i = 0
			
			self.all_data = Data
			self.train_batch = x_train
			self.test_batch = x_test
			self.test_labels = y_test
			self.train_labels = y_train
			self.valid_batch = x_val
			self.valid_labels = y_val
			
		def next_batch(self,batch_size):
			
			batch_x = self.train_batch[self.i:self.i + batch_size]
			batch_y = self.train_labels[self.i:self.i + batch_size]
			self.i = (self.i + batch_size) % len(self.train_batch)
			#print(self.i)
			return batch_x, batch_y 	
			
			
# Initializing values

	A = 4 #Number of filters out, 1st convolution
	B = 8 #Number of filters out, 2nd convolution
	C = 740 #number of weights for flat array

	#Bacth size
	batch_size = 100 #number of spectra given to the neural network at a time

	#Initializing with small random values
	w1 = tf.Variable(tf.truncated_normal([2, 1, A], stddev= 0.1), name="w1")
	b1 = tf.Variable(tf.constant(0.1, tf.float32, [A]), name="b1")
	w2 = tf.Variable(tf.truncated_normal([2, 4, B], stddev= 0.1), name="w2")
	b2 = tf.Variable(tf.constant(0.1, tf.float32, [B]), name="b2")
	w3 = tf.Variable(tf.truncated_normal([74*B, C], stddev= 0.1), name="w3") #79*8 if metabolites
	b3 = tf.Variable(tf.constant(0.1, tf.float32, [C]), name="b3")
	w4 = tf.Variable(tf.truncated_normal([C,len(class_names)], stddev= 0.1), name="w4") 
	b4 = tf.Variable(tf.constant(0.1, tf.float32, [len(class_names)]), name="b4")
	
	# test flag for batch normalization
	tst = tf.placeholder(tf.bool, name="tst")
	iter = tf.placeholder(tf.int32, name="iter")

	# dropout probability
	prob_hold = tf.placeholder(tf.float32, name="prob_hold")
	prob_hold_conv = tf.placeholder(tf.float32, name="prob_hold_conv")

	#For learning rate
	lr = tf.placeholder(tf.float32)

	#For the correct answers
	y_true = tf.placeholder(tf.float32, shape=[None, len(class_names)], name="y_true")


# Model
	x = tf.placeholder(tf.float32, shape=[None, 740], name="input_spectrum")
	x_input = tf.reshape(x, [-1, int(x.shape[1]), 1])

	#with tf.name_scope("Convolutional_layer_1"):
	y1_convo = tf.nn.conv1d(x_input, w1, stride=1, padding='SAME') + b1
	y1_batchnorm, update_ema1 = batchnorm(y1_convo, tst, iter, b1, convolutional=True)
	y1_relu = tf.nn.relu(y1_batchnorm)
	y1 = tf.nn.dropout(y1_relu, prob_hold_conv, compatible_convolutional_noise_shape(y1_relu))
	y1_reshape = tf.reshape(y1, shape=[-1, 1, 740, int(y1.shape[2])])
	pooling_1 = tf.nn.max_pool(y1_reshape, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
	pooling_layer_1 = tf.squeeze(pooling_1,axis=1)

	#with tf.name_scope("Convolutional_layer_2"):
	y2_convo = tf.nn.conv1d(pooling_layer_1, w2, stride=1, padding='SAME') + b2
	y2_batchnorm, update_ema2 = batchnorm(y2_convo, tst, iter, b2, convolutional=True)
	y2_relu = tf.nn.relu(y2_batchnorm)
	y2 = tf.nn.dropout(y2_relu, prob_hold_conv, compatible_convolutional_noise_shape(y2_relu))
	y2_reshape = tf.reshape(y2, shape=[-1, 1, 370, int(y2.shape[2])])
	pooling_2 = tf.nn.max_pool(y2_reshape, ksize=[1,1,5,1], strides=[1,1,5,1], padding='SAME')
	pooling_layer_2 = tf.squeeze(pooling_2,axis=1)

	#with tf.name_scope("Densely_connected_layer_1"):
	y3_flat = tf.reshape(pooling_layer_2, [-1, 74*8])
	y3_linear = tf.matmul(y3_flat, w3) + b3
	y3_batchnorm, update_ema3 = batchnorm(y3_linear, tst, iter, b3)
	y3_relu = tf.nn.relu(y3_batchnorm)
	y3 = tf.nn.dropout(y3_relu, prob_hold)

	#with tf.name_scope("Densely_connected_layer_2"):    
	ylogits = tf.matmul(y3, w4) + b4
	#with tf.name_scope("Softmax"):
	y_pred = tf.nn.softmax(ylogits, name="prediction_op")    

	update_ema = tf.group(update_ema1, update_ema2, update_ema3)
	
# calculation of the loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_true))

#Optimizer
	optimizer = tf.train.AdamOptimizer(lr)
	train = optimizer.minimize(cross_entropy)

	init = tf.global_variables_initializer()

	ch = Opto()

	saver = tf.train.Saver()
	
	# You have to provide the list of all the variable in order to seccessfuly restore the model
	#with the same variables (w,b)
	lst_vars = []
	for v in tf.global_variables():
		lst_vars.append(v)
		#print(v.name, '....')		
		
	saver = tf.train.Saver(var_list=lst_vars)
	
# Start the session:
	
	with tf.Session() as sess:
    sess.run(init)
        
    Acc_Test = []
    Xent_Test = []
    
    for i in range(len(x_train)):        
      
        batch_x, batch_y = ch.next_batch(100)
        
               
        max_learning_rate = 0.02
        min_learning_rate = 0.0001
        decay_speed = 100 + len(x_train) #must be 0.9 or 0.999 of the whole dataset
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        
        _, loss_val = sess.run([train, cross_entropy], feed_dict={x:batch_x, y_true:batch_y, lr:learning_rate, tst: False, prob_hold:0.5, prob_hold_conv: 1.0})
        sess.run(update_ema, feed_dict={x:batch_x, y_true:batch_y, tst: False, iter: i, prob_hold: 1.0, prob_hold_conv: 1.0})
        
                           
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
            
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            print(sess.run(acc,feed_dict={x:ch.test_batch, y_true:ch.test_labels, 
                                          tst: False, prob_hold:1.0, prob_hold_conv:1.0}))
            Acc_Test.append(sess.run(acc,feed_dict={x:ch.test_batch, y_true:ch.test_labels, 
                                          tst: False, prob_hold:1.0, prob_hold_conv:1.0}))
            Xent_Test.append(loss_val)
            
            #Xent.append(sess.run(train, feed_dict={x:batch_x, y_true:batch_y, lr:learning_rate,
                                                           #tst: False, prob_hold:0.5, prob_hold_conv: 1.0})
            
            print('\n')   
            
    save_path = saver.save(sess, "./NAME_OF_YOUR_MODEL.ckpt")
        
    print("Model saved in path: %s" % save_path)
    
    print('\n')
    
	#ensure that the saze model has the right weights
    print(sess.run('w1:0'))
	
	#End if the training
	
###################################################################################################################################

#Apply the model for predictions upon new dataset

	import os
	
	#Extract all your files to predict in the following folder.
	
	def files(path):
		for file in os.listdir(path):
			if os.path.isfile(os.path.join(path, file)):
				yield file
				

	for file in files(r'YOUR_FILE_PATH_TO_PREDICT'):
    
    print(file)
        
    Input = pd.read_csv(r'YOUR_FILE_PATH_TO_PREDICT\%s' %file)
	
	
#Prediction by using the trained model:

	with tf.Session() as sess:
	
		new_saver = tf.train.import_meta_graph('./YOUR_SAVED_MODEL.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./YOUR_SAVED_MODEL_FOLDER/'))
		print(sess.run('w1:0 #ensure that you have the same weight as printed before.
		
		print('\n')
		
		for file in files(r'PATH_TO_YOUR_FILES'):
		   
			x_test = pd.read_csv(r'PATH_TO_YOUR_FILES\%s' %file)
		
			prediction = y_pred.eval(feed_dict={x:x_test, tst:False, prob_hold:1.0, prob_hold_conv:1.0})
			
			Pred = np.asarray(prediction)
			np.savetxt(r'PATH_TO_SAVE\%s' %file + '.csv',
					Pred, delimiter=';')
			
		
		
		
		
		