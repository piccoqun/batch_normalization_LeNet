
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops



class CustomizedBatchNorm(Layer):

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               trainable=True,
               **kwargs):
    super(CustomizedBatchNorm, self).__init__(**kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)

    self.trainable = trainable
    super(CustomizedBatchNorm, self).__init__(**kwargs)


  def build(self, input_shape):
     input_shape = tensor_shape.TensorShape(input_shape)
     ndims = len(input_shape)

     # Convert axis to list and resolve negatives

     self.axis = [self.axis]

     for idx, x in enumerate(self.axis):
       if x < 0:
         self.axis[idx] = ndims + x

     axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}

     # Single axis batch norm (most common/default use-case)
     param_shape = (list(axis_to_dim.values())[0],)

     self.gamma = self.add_weight(
           name='gamma',
           shape=param_shape,
           initializer=self.gamma_initializer,
           trainable=True)

     self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          initializer=self.beta_initializer,
          trainable=True)

     self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=param_shape,
          initializer=self.moving_mean_initializer,
          trainable=False)
    
     self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          initializer=self.moving_variance_initializer,
          trainable=False)

     super(CustomizedBatchNorm, self).build(input_shape)


  # use moving average during training
  def _assign_moving_average(self, variable, value, momentum, inputs_size):
    with K.name_scope('AssignMovingAvg') as scope:
      with ops.colocate_with(variable):
        decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = math_ops.cast(decay, variable.dtype.base_dtype)
        update_delta = (
            variable - math_ops.cast(value, variable.dtype)) * decay
        if inputs_size is not None:
          update_delta = array_ops.where(inputs_size > 0, update_delta,
                                         K.zeros_like(update_delta))
        return state_ops.assign_sub(variable, update_delta, name=scope)

  def _moments(self, inputs, reduction_axes, keep_dims):
    mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)
    return mean, variance

  def call(self, inputs, training=None):
    training = K.learning_phase()

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    scale, offset = self.gamma, self.beta


    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value is not False:

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = len(self.axis) > 1
      mean, variance = self._moments(
          math_ops.cast(inputs, inputs.dtype),
          reduction_axes,
          keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: ops.convert_to_tensor(moving_mean))
      variance = tf_utils.smart_cond(
          training,
          lambda: variance,
          lambda: ops.convert_to_tensor(moving_variance))

      new_mean, new_variance = mean, variance

      if ops.executing_eagerly_outside_functions(
      ) and distribution_strategy_context.has_strategy():
        inputs_size = array_ops.size(inputs)
      else:
        inputs_size = None

      if distribution_strategy_context.in_cross_replica_context():
        strategy = distribution_strategy_context.get_strategy()

        def _do_update(var, value):
          """Compute the updates for mean and variance."""
          return strategy.extended.update(
              var,
              self._assign_moving_average, (value, self.momentum, inputs_size),
              group=False)
        # We need to unwrap the moving_mean or moving_variance in the case of
        # training being false to match the output of true_fn and false_fn
        # in the smart cond.
        def mean_update():
          true_branch = lambda: _do_update(self.moving_mean, new_mean)
          false_branch = lambda: strategy.unwrap(self.moving_mean)
          return tf_utils.smart_cond(training, true_branch, false_branch)

        def variance_update():
          return tf_utils.smart_cond(
              training, lambda: _do_update(self.moving_variance, new_variance),
              lambda: strategy.unwrap(self.moving_variance))
      else:
        def _do_update(var, value):
          """Compute the updates for mean and variance."""
          return self._assign_moving_average(var, value, self.momentum,
                                             inputs_size)


        def mean_update():
          true_branch = lambda: _do_update(self.moving_mean, new_mean)
          false_branch = lambda: self.moving_mean
          return tf_utils.smart_cond(training, true_branch, false_branch)

        def variance_update():
          true_branch = lambda: _do_update(self.moving_variance, new_variance)
          false_branch = lambda: self.moving_variance
          return tf_utils.smart_cond(training, true_branch, false_branch)

      self.add_update(mean_update, inputs=True)
      self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)

    outputs = nn.batch_normalization(inputs,mean,variance,
                                     offset,
                                     scale,
                                     self.epsilon)

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape





