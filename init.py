
import tensorflow as tf

# 初期化(Xavier Initialization)
def xavier_init(size):
    input_dim = size[0]
    xavier_variance = 1. / tf.sqrt(input_dim/2.)
    return tf.random_normal(shape=size, stddev=xavier_variance)
