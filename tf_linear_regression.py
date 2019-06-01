from __future__ import print_function

import tensorflow as tf
import numpy
rng = numpy.random

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))
        
        
        
        
        
        
        
        
        
 结果：
 Epoch: 0050 cost= 0.149805307 W= 0.4005215 b= -0.28429833
Epoch: 0100 cost= 0.141389042 W= 0.39154518 b= -0.21972309
Epoch: 0150 cost= 0.133944899 W= 0.3831026 b= -0.15898821
Epoch: 0200 cost= 0.127360508 W= 0.3751622 b= -0.10186545
Epoch: 0250 cost= 0.121536762 W= 0.3676941 b= -0.04814011
Epoch: 0300 cost= 0.116385765 W= 0.3606701 b= 0.0023900538
Epoch: 0350 cost= 0.111829840 W= 0.35406384 b= 0.049914893
Epoch: 0400 cost= 0.107800238 W= 0.34785056 b= 0.094613194
Epoch: 0450 cost= 0.104236253 W= 0.34200668 b= 0.13665302
Epoch: 0500 cost= 0.101084091 W= 0.33651048 b= 0.17619261
Epoch: 0550 cost= 0.098296180 W= 0.33134103 b= 0.21338077
Epoch: 0600 cost= 0.095830433 W= 0.32647923 b= 0.24835688
Epoch: 0650 cost= 0.093649700 W= 0.32190642 b= 0.2812528
Epoch: 0700 cost= 0.091721006 W= 0.31760564 b= 0.31219286
Epoch: 0750 cost= 0.090015300 W= 0.3135607 b= 0.34129173
Epoch: 0800 cost= 0.088506758 W= 0.30975628 b= 0.3686606
Epoch: 0850 cost= 0.087172642 W= 0.306178 b= 0.39440206
Epoch: 0900 cost= 0.085992813 W= 0.3028127 b= 0.41861176
Epoch: 0950 cost= 0.084949441 W= 0.29964754 b= 0.4413817
Epoch: 1000 cost= 0.084026709 W= 0.29667062 b= 0.46279782
Optimization Finished!
Training cost= 0.08402671 W= 0.29667062 b= 0.46279782 

Testing... (Mean square loss Comparison)
Testing cost= 0.07760731
Absolute mean square loss difference: 0.006419398
