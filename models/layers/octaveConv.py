from tensorflow.layers import AveragePooling2D
from tensorflow.keras.backend import repeat_elements, conv2d
from tensorflow.python.layers import base

class OctaveConv2D(base.Layer):
    def __init__(self, filters, alpha, kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None,
            data_format="channels_last", **kwargs):
        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__()

        self.alpha = alpha
        self.filters = filters

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.data_format = data_format

        # split low and high frequency channels
        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4

        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        self.high2high_kernel = self.add_weight(
                name="high2high_kernel",
                shape=(*self.kernel_size, high_in, self.high_channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        self.high2low_kernel = self.add_weight(
                name="high2low_kernel",
                shape=(*self.kernel_size, high_in, self.low_channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        self.low2high_kernel = self.add_weight(
                name="low2high_kernel",
                shape=(*self.kernel_size, low_in, self.high_channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        self.low2low_kernel = self.add_weight(
                name="low2low_kernel",
                shape=(*self.kernel_size, low_in, self.low_channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        super().build(input_shape)

    def call(self, inputs):
        high_input, low_input = inputs

        # TODO: Rewrite with tensorflow
        high2high = conv2d(high_input, self.high2high_kernel, strides=self.strides,
                padding=self.padding, data_format=self.data_format)

        # down sampling
        high2low = AveragePooling2D((2,2), strides=(2,2), name='high2low_pool2d',)(high_input)
        high2low = conv2d(high2low, self.high2low_kernel, strides=self.strides,
                padding=self.padding, data_format=self.data_format)

        # up sampling
        low2high = conv2d(low_input, self.low2high_kernel, strides=self.strides,
                padding=self.padding, data_format=self.data_format)
        low2high = repeat_elements(low2high, 2, axis=1)
        low2high = repeat_elements(low2high, 2, axis=2)

        low2low = conv2d(low_input, self.low2low_kernel, strides=self.strides,
                padding=self.padding, data_format=self.data_format)

        high_add = high2high + low2high
        low_add = high2low + low2low
        return [high_add, low_add]

    def compute_output_shape(self, input_shape):
        high_in_shapoe, low_in_shape = input_shape
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
                **base_config,
                "filters": self.filters,
                "alpha": self.alpha,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_constraint": self.kernel_constraint,
                }
        return out_config
