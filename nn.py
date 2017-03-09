import tensorflow as tf
import numpy as np

# 07train.txt
# # x1 x2 y
# 0   0   0
# 0   1   1
# 1   0   1
# 1   1   0

xy = np.loadtxt('07train.txt', unpack=True)


############

# x_data = xy[:-1]
# y_data = xy[-1]
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))
#
# h = tf.matmul(W, X)
# hypothesis = tf.div(1., 1. + tf.exp(-h))

############

x_data = np.transpose(xy[:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

############

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 999:
            # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1,W2]))
            r1, (r2, r3) = sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2])
            print('{:5} {:10.8f} {} {}'.format(step + 1, r1, np.reshape(r2, (1, 4)), np.reshape(r3, (1, 2))))

    print('-'*50)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    #Calculate accuraty
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    param = [hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy]
    result = sess.run(param, feed_dict={X:x_data, Y:y_data})

    for i in result:
        print(i)
    print('Accuracy :', accuracy.eval({X:x_data, Y:y_data}))

