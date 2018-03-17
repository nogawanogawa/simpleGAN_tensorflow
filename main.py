import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from init import xavier_init
from img_gen import plot

# generatorの入力となる乱数
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

#　generatorのパラメータの初期化
G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

#　discriminator用の教師データのplaceholder [28x28=784]
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

# discriminatorのパラメータの初期化
D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# generatorのセットアップ
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

# discriminatorのセットアップ
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# Flowのセットアップ
G_sample = generator(Z)

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# Lossの計算のセットアップ
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))


# discriminatorのパラメータの更新
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

# generatorのパラメータの更新
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# iteration 1000回毎にサンプル画像を生成するときのgeneratorの入力を生成
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

batch_size = 128
Z_dim = 100

# sessionの定義
sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

# 画像生成用ディレクトリの作成
if not os.path.exists('output/'):
    os.makedirs('output/')

#　画像ファイルの通番初期化
i = 0

# 学習プロセス開始
for itr in range(10000):
    if itr % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('output/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    if itr % 1000 == 0:
        print('Iter: {}'.format(itr))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
