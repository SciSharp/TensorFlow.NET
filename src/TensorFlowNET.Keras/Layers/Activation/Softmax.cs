﻿using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Softmax : Layer {
            Axis axis;
            public Softmax ( SoftmaxArgs args ) : base(args) {
                  axis = args.axis;
            }
<<<<<<< HEAD
<<<<<<< HEAD
            protected override Tensors Call ( Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
                  Tensor x = inputs.Length == 2 ? inputs[0] + ((1.0 - tf.cast(inputs[1], inputs.dtype)) * 1e-9)
=======
            protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
            {
                  Tensor x = inputs.Length == 2 ? inputs + ((1.0 - tf.cast(inputs[1], inputs.dtype)) * 1e-9)
>>>>>>> master
=======
            protected override Tensors Call ( Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
                  Tensor x = inputs.Length == 2 ? inputs[0] + ((1.0 - tf.cast(inputs[1], inputs.dtype)) * 1e-9)
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
                                                : inputs;
                  Tensor e = tf.exp(tf.sub(x, tf.reduce_max(x, axis: this.axis, keepdims: true)));
                  Tensor s = tf.reduce_sum(e, axis: this.axis, keepdims: true);
                  return tf.div(e, s);
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
