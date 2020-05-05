from tensorflow.keras.layers import Layer
import tensorflow as tf


class ChannelPadder(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ChannelPadder, self).__init__(**kwargs)

    def transform_dimensions(self, dm, output_dim):
        result = dm
        nc = dm.shape[2]
        if nc is not None:
            if nc < output_dim:
                for i in range(0, output_dim - 2 * nc - 2, nc):
                    paddings = [[0, 0], [0, 0], [0, nc]]
                    result = tf.pad(result, paddings, 'SYMMETRIC')
                paddings = [[0, 0], [0, 0], [0, output_dim - result.shape[2]]]
                result = tf.pad(result, paddings, 'SYMMETRIC')
        return result

    def call(self, X, **kwargs):
        return self.transform_dimensions(X, self.output_dim)