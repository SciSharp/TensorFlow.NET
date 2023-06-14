﻿using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Softplus : Layer {
            public Softplus ( LayerArgs args ) : base(args) {
                  // Softplus has no arguments
            }
<<<<<<< HEAD
<<<<<<< HEAD
            protected override Tensors Call ( Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
=======
            protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
            {
>>>>>>> master
=======
            protected override Tensors Call ( Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
                  Tensor x = inputs;
                  return tf.log(
                        tf.add(tf.exp(x), 1f));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
