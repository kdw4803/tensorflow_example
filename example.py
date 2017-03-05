import tensorflow as tf

def hello():
    a = tf.constant('hello, tensorflow!')
    print(a)  # Tensor("Const:0", shape=(), dtype=string)

    sess = tf.Session()
    result = sess.run(a)

    # 2.x 버전에서는 문자열로 출력되지만, 3.x 버전에서는 byte 자료형
    # 문자열로 변환하기 위해 decode 함수로 변환
    print(result)  # b'hello, tensorflow!'
    print(type(result))  # <class 'bytes'>
    print(result.decode(encoding='utf-8'))  # hello, tensorflow!
    print(type(result.decode(encoding='utf-8')))  # <class 'str'>

    # 세션 닫기
    sess.close()

# hello();


def constant():
    a = tf.constant(2)
    b = tf.constant(3)

    # with 구문을 벗어날 때, 종료 코드가 있다면 대신 호출해 줌
    # 예외가 발생한 경우에도 보장
    with tf.Session() as sess:
        result = sess.run(a+b)
        print(type(result))             # <class 'numpy.int32'>
        print(result)                   # 5

        # int 자료형과 연산 가능
        print(result + 7)               # 12
        print(type(result + 7))         # <class 'numpy.int64'>

# constant();



def placeHolder():
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    add = tf.add(a, b)
    mul = tf.mul(a, b)

    with tf.Session() as sess:
        # {a: 2, b: 3}는 딕셔너리
        # key로 'a'와 'b'를 사용하고, value로 2와 3  사용
        # free_dict를 사용하지 않을 경우 None 기본값 적용
        r1 = sess.run(add, feed_dict={a: 2, b: 3})
        r2 = sess.run(mul, feed_dict={a: 2, b: 3})

        print(type(r1))                 # <class 'numpy.int16'>
        print(r1, r2)                   # 5, 6

# placeHolder();



def showTensor():
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    # x에 대해서 연산을 수행해서 결과를 먼저 만든다.
    x.initializer.run()     # Initialize 'x' using the run() method of its initializer op.

    sub = tf.sub(x, a)      # Add an op to subtract 'a' from 'x'.  Run it and print the result
    print(sub.eval())       # [-2. -1.]

    print('-------------------------------------')

    # 결과를 내장하고 있다면 eval() 사용 가능. initializer 없이 x에 대해서 호출하면 비정상 종료
    print(a.eval())         # [ 3.  3.]
    print(x.eval())         # [ 1.  2.]

    # -1에서 1 사이의 정규분포 난수 3개 생성. b는 1행 3열의 텐서 객체
    b = tf.random_uniform([3], -1.0, 1.0)
    print(type(b))          # <class 'tensorflow.python.framework.ops.Tensor'>
    print(b.eval())         # [-0.16271138 -0.33350062  0.51194   ]

    # tensor라면 initializer 사용
    w = tf.Variable(tf.random_uniform([5, 3], 0, 32, dtype=tf.int32))
    w.initializer.run()
    print(w.eval())         # [[15  1 21] [14 16 27] [13 30 28] [23 21 26] [15 19 16]]

    print('-------------------------------------')

    x = [[1., 1.], [10., 2.]]
    print(tf.reduce_mean(x).eval())         # 3.5, 전체 평균
    print(tf.reduce_mean(x, 0).eval())      # [ 5.5  1.5], 0은 column
    print(tf.reduce_mean(x, 1).eval())      # [ 1.  6.], 1은 row

    sess.close()


showTensor();

