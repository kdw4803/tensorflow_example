# import tensorflow as tf

# x_data = [1., 2., 3.]
# y_data = [1., 2., 3.]
#
# # try to find values for w and b that compute y_data = W * x_data + b
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#
# # my hypothesis
# hypothesis = W * x_data + b
#
# # Simplified cost function
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
# # minimize
# rate = tf.Variable(0.1)  # learning rate, alpha
# optimizer = tf.train.GradientDescentOptimizer(rate)
# train = optimizer.minimize(cost)
#
# # before starting, initialize the variables. We will 'run' this first.
# init = tf.global_variables_initializer()
#
# # launch the graph
# sess = tf.Session()
# sess.run(init)
#
# # fit the line
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))
#
# # learns best fit is W: [1] b: [0]



#################### placeholder ######################


# x_data = [1., 2., 3., 4.]
# y_data = [2., 4., 6., 8.]
#
# # range is -100 ~ 100
# W = tf.Variable(tf.random_uniform([1], -100., 100.))
# b = tf.Variable(tf.random_uniform([1], -100., 100.))
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# hypothesis = W * X + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# rate = tf.Variable(0.1)
# optimizer = tf.train.GradientDescentOptimizer(rate)
# train = optimizer.minimize(cost)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(2001):
#     sess.run(train, feed_dict={X: x_data, Y: y_data})
#     if step % 20 == 0:
#         print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
#
# print(sess.run(hypothesis, feed_dict={X: 5}))           # [ 10.]
# print(sess.run(hypothesis, feed_dict={X: 2.5}))         # [5.]
# print(sess.run(hypothesis, feed_dict={X: [2.5, 5]}))    # [  5.  10.], 원하는 X의 값만큼 전달.



############## graph ###############


# X = [1., 2., 3.]
# Y = [1., 2., 3.]
# m = len(X)
#
# W = tf.placeholder(tf.float32)
#
# hypothesis = tf.mul(W, X)
# cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / m
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# # 그래프로 표시하기 위해 데이터를 누적할 리스트
# W_val, cost_val = [], []
#
# # 0.1 단위로 증가할 수 없어서 -30부터 시작. 그래프에는 -3에서 5까지 표시됨.
# for i in range(-30, 51):
#     xPos = i*0.1                                    # x 좌표. -3에서 5까지 0.1씩 증가
#     yPos = sess.run(cost, feed_dict={W: xPos})      # x 좌표에 따른 y 값
#
#     print('{:3.1f}, {:3.1f}'.format(xPos, yPos))
#
#     # 그래프에 표시할 데이터 누적. 단순히 리스트에 갯수를 늘려나감
#     W_val.append(xPos)
#     cost_val.append(yPos)
#
# sess.close()
#
# # ------------------------------------------ #
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.plot(W_val, cost_val, 'bo')
# plt.ylabel('Cost')
# plt.xlabel('W')
# plt.show()


import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]           # x와 y의 관계가 모호하다. cost가 내려가지 않는 것이 맞을 수도 있다.

# 동영상에 나온 데이터셋. 40번째 위치에서 best fit을 찾는다. 이번에는 사용하지 않음.
# x_data = [1., 2., 3.]
# y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10000., 10000.))        # tensor 객체 반환

X = tf.placeholder(tf.float32)      # 반복문에서 x_data, y_data로 치환됨
Y = tf.placeholder(tf.float32)

hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 동영상에서 미분을 적용해서 구한 새로운 공식. cost를 계산하는 공식
mean    = tf.reduce_mean(tf.mul(tf.mul(W, X) - Y, X))   # 변경된 W가 mean에도 영향을 준다
descent = W - tf.mul(0.1, mean)
# W 업데이트. tf.assign(W, descent). 호출할 때마다 변경된 W의 값이 반영되기 때문에 업데이트된다.
update  = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(50):
    uResult = sess.run(update, feed_dict={X: x_data, Y: y_data})    # 이 코드를 호출하지 않으면 W가 바뀌지 않는다.
    cResult = sess.run(  cost, feed_dict={X: x_data, Y: y_data})    # update에서 바꾼 W가 영향을 주기 때문에 같은 값이 나온다.
    wResult = sess.run(W)
    mResult = sess.run(mean, feed_dict={X: x_data, Y: y_data})

    # 결과가 오른쪽과 왼쪽 경사를 번갈아 이동하면서 내려온다. 기존에 한 쪽 경계만 타고 내려오는 것과 차이가 있다.
    # 최종적으로 오른쪽과 왼쪽 경사의 중앙에서 최소 비용을 얻게 된다. (생성된 난수값에 따라 한쪽 경사만 타기도 한다.)
    # descent 계산에서 0.1 대신 0.01을 사용하면 오른쪽 경사만 타고 내려오는 것을 확인할 수 있다. 결국 step이 너무 커서 발생한 현상
    print('{} {} {} [{}, {}]'.format(step, mResult, cResult, wResult, uResult))

print('-'*50)
print('[] 안에 들어간 2개의 결과가 동일하다. 즉, update와 cost 계산값이 동일하다.')

print(sess.run(hypothesis, feed_dict={X: 5.0}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

sess.close()



