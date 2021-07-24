﻿using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Softplus : Layer {
            public Softplus ( LayerArgs args ) : base(args) {
                  // Softplus has no arguments
            }
            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor x = inputs;
                  return tf.log(
                        tf.add(tf.exp(x), 1f));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
