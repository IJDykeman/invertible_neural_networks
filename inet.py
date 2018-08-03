import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import uuid



def split(x):
    half_x_dim = x.shape[1] // 2
    return x[:, :half_x_dim], x[:, half_x_dim:]

def fuse(x1, x2):
    return tf.concat([x1, x2], -1)


class HalfChannelsRotator:
    def rotate(self, x):
        """
        This method could be replaced with shuffling (as in Real NVP) or with a learned rotation, as in GLOW.
        This rotation by reversal of feature order is used in NICE
        """
        return x[:,::-1]

    def rotate_inverse(self, x):
        return x[:,::-1]

class DenseLayer:
    def __init__(self, x, rotator, unique_name=""):
        # we need to ensure that someone does not accidentalyl end up with 2 layers with the same name
        self.rotator = rotator
        self.unique_name = unique_name + uuid.uuid4().hex.upper()[0:6] 
        half_x_dim = x.get_shape()[1].value // 2
        x_rotated = self.rotator.rotate(x)
        x1, x2 = split(x_rotated)
        scale, translate = self.get_transform(x1)
        x2_transformed = (x2 * scale) + translate
        self.forward_output = fuse(x1, x2_transformed)
        
        # real NVP eq 6
        self.determinant = 1 * half_x_dim + tf.reduce_sum(scale, axis=-1)

    def get_transform(self, x):
        input_dim = x.get_shape()[1].value
        with tf.variable_scope(self.unique_name, reuse=tf.AUTO_REUSE):
            h1 = slim.fully_connected(x, input_dim + 5, activation_fn=tf.nn.elu) # +5 just to show that these can be any size
            h2 = slim.fully_connected(h1, input_dim + 5, activation_fn=tf.nn.elu) # +5 just to show that these can be any size
            h3 = slim.fully_connected(h2, input_dim, activation_fn=tf.nn.elu)
            scale = tf.exp(slim.fully_connected(h3, input_dim, activation_fn=tf.nn.elu))
            translate = slim.fully_connected(h3, input_dim)
        return scale, translate

    def reverse(self, y):
        x1, x2_transformed = split(y)
        scale, translate = self.get_transform(x1)
        x2 = (x2_transformed - translate) / scale
        x_rotated = tf.concat([x1, x2], -1)
        x = self.rotator.rotate_inverse(x_rotated)
        self.reverse_output = x

for i in range(1000):
    x = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    layer = DenseLayer(x, HalfChannelsRotator(), unique_name="layer1")
    y = layer.forward_output
    layer.reverse(y)
    x_recovered = layer.reverse_output
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # show that the inverse is correct
    assert np.allclose(x.eval(session=sess), x_recovered.eval(session=sess))
    print i
