import tensorflow as tf 
import numpy as np 
import dataprep

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, shape = [None,100])

outputs = tf.placeholder(tf.float32, shape = [None,1])

first_hidden = 4 

w1 = tf.Variable(tf.truncated_normal([100,first_hidden]))

b1 = tf.Variable(tf.zeros([first_hidden]))

layer_1_output = tf.nn.sigmoid(tf.matmul(inputs,w1) + b1 )

w2 = tf.Variable(tf.truncated_normal([first_hidden,100]))

b2 = tf.Variable(tf.zeros([100]))

layer_2_output = tf.nn.sigmoid(tf.matmul(layer_1_output,w2) + b2)

w3 = tf.Variable(tf.truncated_normal([100,1]))

b3 = tf.Variable(tf.zeros([1]))

result = tf.nn.sigmoid(tf.matmul(layer_2_output,w3) + b3 )

error = 0.5*tf.reduce_sum(tf.subtract(result,outputs) * tf.subtract(result,outputs))

train_fixer = tf.train.GradientDescentOptimizer(0.05).minimize(error)            

sess.run(tf.initialize_all_variables())


#training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


training_inputs = [dataprep.v1,dataprep.v2]



output1 = [[0.0], [1.0]]
print ("OUTPUT1")
print (output1)

for i in range(0,3):
    _,loss = sess.run([train_fixer, error],feed_dict =
     {inputs:np.array(training_inputs),outputs:np.array(output1)})
    print (loss)