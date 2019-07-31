""" Extensions to TF RNN class by una_dinosaria"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv

if pv(tf.__version__) >= pv('1.2.0'):
    from tensorflow.contrib.rnn import LSTMStateTuple
else:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv

from tensorflow.python.ops import variable_scope as vs

import collections
import math


class ResidualWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell):
        """Create a cell with added residual connection.

        Args:
          cell: an RNNCell. The input is added to the output.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Add the residual connection
        output = tf.add(output, inputs)

        return output, new_state


class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print(' state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):

            # Fine if GRU...
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            if isinstance(insize, LSTMStateTuple):
                insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        self.w_out = tf.get_variable("proj_w_out",
                                     [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


class LinearSpaceSpikingDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print(' state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):

            # Fine if GRU...
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            if isinstance(insize, LSTMStateTuple):
                insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        # self.w_out = tf.get_variable("proj_w_out",
        #                              [insize, output_size],
        #                              dtype=tf.float32,
        #                              initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        # self.b_out = tf.get_variable("proj_b_out", [output_size],
        #                              dtype=tf.float32,
        #                              initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        #
        # self.linear_output_size = output_size
        self.insize = insize
        self.kernel = tf.get_variable('spiking_kernel',
                                      [output_size, insize],
                                      dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.bias = tf.get_variable('spiking_bias',
                                    [insize],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.weights_past = tf.get_variable('weights_past',
                                            [insize],
                                            dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.w_out = tf.get_variable("proj_w_out",
                                     [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.weights_tao = tf.get_variable('weights_tao',
                                          [insize],
                                          dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        # output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        gate = tf.tanh(tf.add(tf.matmul(inputs, self.kernel), self.bias))
        # gate = tf.tanh(tf.add(tf.matmul(inputs, self.kernel), self.bias))

        lag = self.insize
        tao_1 = tf.constant(50, 'float32')
        tem1 = tf.cast(np.linspace(1, lag, num=lag, endpoint=True, dtype='float32'), dtype=tf.float32)
        tem2 = tf.cast(np.zeros(lag) + (lag + 3), dtype=tf.float32)

        sj1 = tf.subtract(tem2, tem1)  # vector length lag
        k1 = tf.divide(-sj1, tao_1)
        ibsino = tf.multiply(self.weights_tao, tf.exp(k1))

        output = tf.multiply(gate, ibsino)  # dot multiply

        output = tf.matmul(output, self.w_out) + self.b_out
        return output, output


# class LinearSpaceTraditionalSpikingDecoderWrapper(RNNCell):
#     """Operator adding a linear encoder to an RNN cell"""
#
#     def __init__(self, cell, output_size):
#         """Create a cell with with a linear encoder in space.
#
#         Args:
#           cell: an RNNCell. The input is passed through a linear layer.
#
#         Raises:
#           TypeError: if cell is not an RNNCell.
#         """
#         if not isinstance(cell, RNNCell):
#             raise TypeError("The parameter cell is not a RNNCell.")
#
#         self._cell = cell
#
#         print('output_size = {0}'.format(output_size))
#         print(' state_size = {0}'.format(self._cell.state_size))
#
#         # Tuple if multi-rnn
#         if isinstance(self._cell.state_size, tuple):
#
#             # Fine if GRU...
#             insize = self._cell.state_size[-1]
#
#             # LSTMStateTuple if LSTM
#             if isinstance(insize, LSTMStateTuple):
#                 insize = insize.h
#
#         else:
#             # Fine if not multi-rnn
#             insize = self._cell.state_size
#
#         # self.w_out = tf.get_variable("proj_w_out",
#         #                              [insize, output_size],
#         #                              dtype=tf.float32,
#         #                              initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
#         # self.b_out = tf.get_variable("proj_b_out", [output_size],
#         #                              dtype=tf.float32,
#         #                              initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
#         #
#         # self.linear_output_size = output_size
#         self.insize = insize
#         self.net_params = SlayerParams("parametersEEG.yaml")
#         self.trainer = SlayerTrainer(self.net_params)
#         # self.dynamic_des_spike = self.net_params['dynamic_des_spike']
#
#         self.input_srm = self.trainer.calculate_srm_kernel()
#         self.srm = self.trainer.calculate_srm_kernel(1)
#         self.ref = self.trainer.calculate_ref_kernel()
#
#         self.fc1 = SpikeLinear(self.net_params['input_x'] * self.net_params['input_y'] * self.net_params['input_channels'],
#                                self.net_params['output_dim'])
#         nn.init.normal_(self.fc1.weight, mean=0.02, std=0.01)
#
#         self.linear_output_size = output_size
#
#     @property
#     def state_size(self):
#         return self._cell.state_size
#
#     @property
#     def output_size(self):
#         return self.linear_output_size
#
#     def __call__(self, inputs, state, scope=None):
#         """Use a linear layer and pass the output to the cell."""
#
#         # Run the rnn as usual
#         x = self.trainer.apply_srm_kernel(inputs, self.input_srm)
#         # Flatten the array
#         x = x.reshape((self.net_params['batch_size'], 1, 1,
#                        self.net_params['input_x'] * self.net_params['input_y'] * self.net_params['input_channels'], -1))
#         # # Linear + activation
#         x1 = self.fc1(x)
#         # x = SpikeFunc.apply(x, self.net_params, self.ref, self.net_params['af_params']['sigma'][0], self.device)
#         # x1= SpikeFunc_MST.apply(x, self.net_params, self.ref, self.net_params['af_params']['sigma'][2], 0, self.device, des_spikes, train_mode, Num_epoch, save_figure)
#
#         x = SpikeFunc_MST.apply(x1, inputs, self.net_params, self.ref, device, train_mode, des_spikes)
#
#         return x
#         return output, output
