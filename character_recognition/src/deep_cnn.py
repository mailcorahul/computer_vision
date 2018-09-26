import tensorflow as tf
import numpy as np
import os
import re
import cv2
import time
import sys
from heapq import merge
np.set_printoptions(threshold=np.nan)

#from stn import trans

#deep_cnn_model_path = "/home/zlabs/zlabsnn/tomcat/apache-tomcat-7.0.75/apache-tomcat-7.0.75/webapps/test/python/ocr/deep_cnn_models/train3_max_4.ckpt"
deep_cnn_model_path = "/home/zlabs/zlabsnn/tomcat/apache-tomcat-7.0.75/apache-tomcat-7.0.75/webapps/test/python/ocr/deep_cnn_models/new_single_gpu_max_accuracy_v1.ckpt"
#deep_cnn_model_path = "/home/zlabs/zlabsnn/tomcat/apache-tomcat-7.0.75/apache-tomcat-7.0.75/webapps/test/python/ocr/deep_cnn_models/73_new_deep_single_gpu_max_accuracy.ckpt"
n_classes = 94

trans_output = sys.argv[1] ;



test_data = np.load( trans_output )
#test_labels = np.zeros((test_data.shape[0],94))

print test_data.shape



# CNN - Recognizer

def xavier_initializer(num, shape) :
     return tf.get_variable(str(num),shape=shape , initializer = tf.contrib.layers.xavier_initializer()) ;

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

test_data  = np.reshape(test_data, (-1, 28, 28, 1))
print test_data.shape

img_size = 28

learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
x = tf.reshape(x, shape=[-1, 28, 28, 1])

# RECOGNIZER
val = 4
fs3 = 3
fs5 = 5

nf1  = 128
nf2  = 256
nf3  = 256

nf4  = 256
nf5  = 256
nf6  = 256

nf7  = 512
nf8  = 256
nf9  = 512

nf10 = 512
nf11 = 256
nf12 = 512  

nf13 = 256
nf14 = 256
nf15 = 256

nfc1 = 4096
nfc2 = 4096
nfc3 = 4096



W_conv1_1 = xavier_initializer( 1, [fs3, fs3,   1, nf1])
W_conv1_2 = xavier_initializer( 2, [fs3, fs3, nf1, nf2])
W_conv1_3 = xavier_initializer( 3, [fs3, fs3, nf2, nf3])

W_conv2_1 = xavier_initializer( 4, [fs5, fs5, nf3, nf4])
W_conv2_2 = xavier_initializer( 5, [fs3, fs3, nf4, nf5])
W_conv2_3 = xavier_initializer( 6, [fs3, fs3, nf5, nf6])

W_conv3_1 = xavier_initializer( 7, [fs3, fs3, nf6, nf7])
W_conv3_2 = xavier_initializer( 8, [fs3, fs3, nf7, nf8])
W_conv3_3 = xavier_initializer( 9, [fs3, fs3, nf8, nf9])

W_conv4_1 = xavier_initializer( 10, [fs3, fs3, nf9,  nf10])
W_conv4_2 = xavier_initializer( 11, [fs3, fs3, nf10, nf11])
W_conv4_3 = xavier_initializer( 12, [fs3, fs3, nf11, nf12])

W_conv5_1 = xavier_initializer( 13, [fs3, fs3, nf12, nf13])
W_conv5_2 = xavier_initializer( 14, [fs3, fs3, nf13, nf14])
W_conv5_3 = xavier_initializer( 15, [fs3, fs3, nf14, nf15])

W_fc1     = xavier_initializer( 16, [img_size//val * img_size//val * nf15, nfc1])
W_fc2     = xavier_initializer( 17, [nfc1, nfc2])
W_fc3     = xavier_initializer( 18, [nfc2, nfc3])
W_fc4     = xavier_initializer( 19, [nfc3, n_classes])

b_conv1_1 = xavier_initializer( 20, [nf1])
b_conv1_2 = xavier_initializer( 21, [nf2])
b_conv1_3 = xavier_initializer( 22, [nf3])

b_conv2_1 = xavier_initializer( 23, [nf4])
b_conv2_2 = xavier_initializer( 24, [nf5])
b_conv2_3 = xavier_initializer( 25, [nf6])

b_conv3_1 = xavier_initializer( 26, [nf7])
b_conv3_2 = xavier_initializer( 27, [nf8])
b_conv3_3 = xavier_initializer( 28, [nf9])

b_conv4_1 = xavier_initializer( 29, [nf10])
b_conv4_2 = xavier_initializer( 30, [nf11])
b_conv4_3 = xavier_initializer( 31, [nf12])

b_conv5_1 = xavier_initializer( 32, [nf13])
b_conv5_2 = xavier_initializer( 33, [nf14])
b_conv5_3 = xavier_initializer( 34, [nf15])

b_fc1     = xavier_initializer( 35, [nfc1])
b_fc2     = xavier_initializer( 36, [nfc2])
b_fc3     = xavier_initializer( 37, [nfc3])
b_fc4     = xavier_initializer( 38, [n_classes])





h_conv1_1 = tf.nn.relu(tf.nn.conv2d(input=x        , filter=W_conv1_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1)
h_conv1_2 = tf.nn.relu(tf.nn.conv2d(input=h_conv1_1, filter=W_conv1_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_2)
h_conv1_3 = tf.nn.relu(tf.nn.conv2d(input=h_conv1_2, filter=W_conv1_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_3)
h_pool1 = tf.nn.max_pool(h_conv1_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

h_conv2_1 = tf.nn.relu(tf.nn.conv2d(input=h_pool1  , filter=W_conv2_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1)
h_conv2_2 = tf.nn.relu(tf.nn.conv2d(input=h_conv2_1, filter=W_conv2_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_2)
h_conv2_3 = tf.nn.relu(tf.nn.conv2d(input=h_conv2_2, filter=W_conv2_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_3)
h_pool2 = tf.nn.max_pool(h_conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

h_conv3_1 = tf.nn.relu(tf.nn.conv2d(input=h_pool2  , filter=W_conv3_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_1)
h_conv3_2 = tf.nn.relu(tf.nn.conv2d(input=h_conv3_1, filter=W_conv3_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_2)
h_conv3_3 = tf.nn.relu(tf.nn.conv2d(input=h_conv3_2, filter=W_conv3_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_3)

h_conv4_1 = tf.nn.relu(tf.nn.conv2d(input=h_conv3_3, filter=W_conv4_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv4_1)
h_conv4_2 = tf.nn.relu(tf.nn.conv2d(input=h_conv4_1, filter=W_conv4_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv4_2)
h_conv4_3 = tf.nn.relu(tf.nn.conv2d(input=h_conv4_2, filter=W_conv4_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv4_3)

h_conv5_1 = tf.nn.relu(tf.nn.conv2d(input=h_conv4_3, filter=W_conv5_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv5_1)
h_conv5_2 = tf.nn.relu(tf.nn.conv2d(input=h_conv5_1, filter=W_conv5_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv5_2)
h_conv5_3 = tf.nn.relu(tf.nn.conv2d(input=h_conv5_2, filter=W_conv5_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv5_3)


#h_pool3 = tf.nn.max_pool(h_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')




h_conv5_flat = tf.reshape(h_conv5_3, [-1, img_size//val * img_size//val * nf15])

h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1), keep_prob)
h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1,        W_fc2) + b_fc2), keep_prob)
h_fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc2,        W_fc3) + b_fc3), keep_prob)
y_logits = tf.matmul(h_fc3, W_fc4) + b_fc4

oo = tf.argmax(y_logits, 1)

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_logits, y) )
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

saver.restore(sess, deep_cnn_model_path);

BATCH_SIZE = 100
"""
test_iter_per_epoch = 1 + (test_data.shape[0]/batch_size)
print test_iter_per_epoch
test_indices = np.linspace(0, test_data.shape[0] - 1, test_iter_per_epoch)
test_indices = test_indices.astype('int')
for test_iter_i in range(test_iter_per_epoch - 1):
    test_batch_xs = test_data[test_indices[test_iter_i]:test_indices[test_iter_i+1], :]
    test_batch_ys = test_labels[test_indices[test_iter_i]:test_indices[test_iter_i+1]]
    CD_output.append(sess.run(oo,feed_dict={x: test_batch_xs, y: test_batch_ys,keep_prob: 1.0}))
"""
print(test_data.shape)
NUM_BATCHES = int(np.ceil(float(test_data.shape[0])/ BATCH_SIZE))
print("\nNO OF BATCHES ---- ",NUM_BATCHES,"\n")
CD_output = []
for k in range(NUM_BATCHES):
	if k < NUM_BATCHES-1 :
	        batch_data = test_data[ k * BATCH_SIZE : (k+1) * BATCH_SIZE, :]
	else : 
		batch_data = test_data[ k * BATCH_SIZE : , :]
	#CD_output += list(sess.run(oo,feed_dict={x: batch_data, y: batch_labels,keep_prob: 1.0}))
        CD_output += list(sess.run(oo,feed_dict={x: batch_data,keep_prob: 1.0}))




#CD_output = sess.run(oo,feed_dict={x: test_data,y: test_labels,keep_prob: 1.0})

if n_classes == 94 :
	alpha = [ ['0','1','2','3','4','5','6','7','8','9'] ,
          ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] ,
          ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] ,
          ["&" ,"%" ,"@" , "/","#",",","$",".","*","=","-", "`","~","!","^","(",")","+","{","}","[","]","|","<",">","?",":",";","'","\\","\"","_"]] ;
	alpha_ii=[[48,48,48,48,48,48,48,48,48,48],
          [87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87],
          [29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29],
          [24,26,0,18,31,23,32,23,28,10,27,-23,-52,42,-18,37,37,36,-43,-44,-9,-10,-40,25,24,24,30,30,51,-1,58,-2]
        ];
if n_classes == 73 :
	alpha = [ ['0','1','2','3','4','5','6','7','8','9'] ,
          ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"] ,
          ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"] ,
          ["&","*","@",",","$",".","=","#","-","%","/"]
        ];
	alpha_ii=[[48,48,48,48,48,48,48,48,48,48],
          [87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87],
          [29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29],
          [24,21,0,21,30,21,7,34,25,34,25]
        ];

CD_output = np.array(CD_output)
OUT = "";
for iter in range(CD_output.shape[0]) :
  for i in range(4):
    for j in range(len(alpha[i])) :
      if i == 3 :
        if CD_output[iter] == ord(alpha[i][j])+alpha_ii[i][j] :
          OUT += alpha[i][j];
      else : 
        if CD_output[iter] == ord(alpha[i][j])-alpha_ii[i][j] :
          OUT += alpha[i][j];

word = np.load(sys.argv[3] + 'word.npy') ;
line = np.load(sys.argv[3] + 'line.npy') ;

word = list(merge(word, line))


def insert_space(string, index):
    return string[:index] + ' ' + string[index:]
def insert_newline(string, index):
    return string[:index] + '\n' + string[index:]

temp = OUT


index = 0
count = 1
pointer = 0

for i in range(len(word)):
	if word[i] == line[pointer] :
		#print count+index+word[i],"linw"
		temp = insert_newline(temp,count+index+word[i])
		pointer += 1
		count += 1
	else :
		#print word[i],"space"
		#print count+index+word[i],"space"
		temp = insert_space(temp,count+index+word[i])
		count += 1

print len(temp);
print temp 
k=os.path.join(sys.argv[2],'cnn');
if not os.path.exists(k):
	os.makedirs(k);	
file = open( sys.argv[2] + "/cnn/cnn_output.txt","w+")
file.write(temp)
file.close()

