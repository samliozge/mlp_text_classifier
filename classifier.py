import tensorflow as tf 
import numpy as np 
import dataprep

class Classifier:
    
    def __init__(self,first_hidden=4):

        self.sess = tf.InteractiveSession()

        self.inputs = tf.placeholder(tf.float32, shape = [None,100])

        self.outputs = tf.placeholder(tf.float32, shape = [None,1])

        self.first_hidden = first_hidden

        self.w1 = tf.Variable(tf.truncated_normal([100,self.first_hidden]))

        self.b1 = tf.Variable(tf.zeros([self.first_hidden]))

        self.layer_1_output = tf.nn.sigmoid(tf.matmul(self.inputs,self.w1) + self.b1 )

        self.w2 = tf.Variable(tf.truncated_normal([self.first_hidden,100]))

        self.b2 = tf.Variable(tf.zeros([100]))

        self.layer_2_output = tf.nn.sigmoid(tf.matmul(self.layer_1_output,self.w2) + self.b2)

        self.w3 = tf.Variable(tf.truncated_normal([100,1]))

        self.b3 = tf.Variable(tf.zeros([1]))
    
    def error_correction(self):

        result = tf.nn.sigmoid(tf.matmul(self.layer_2_output,self.w3) + self.b3 )

        self.error = 0.5*tf.reduce_sum(tf.subtract(result,self.outputs) * tf.subtract(result,self.outputs))

        self.train_fixer = tf.train.GradientDescentOptimizer(0.05).minimize(self.error)            

        self.sess.run(tf.initialize_all_variables())
    
    def trainNN(self,x,y):

       for i in range(0,3):
            _,loss = self.sess.run([self.train_fixer, self.error],feed_dict =
            {self.inputs:np.array(x),self.outputs:np.array(y)})
       print (loss)


    
model = Classifier()
model.error_correction()
       
x = [dataprep.v1,dataprep.v2]
y = [[0.0], [1.0]]
model.trainNN(x,y)