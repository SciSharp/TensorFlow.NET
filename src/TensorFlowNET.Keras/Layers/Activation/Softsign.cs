using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Softsign : Layer {
            public Softsign ( LayerArgs args ) : base(args) {
                  // Softsign has no arguments
            }
            protected override Tensors Call ( Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
                  Tensor x = inputs;
                  // x / (abs(x) + 1)
                  return tf.div(x, tf.add(1f, tf.abs(x)));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
