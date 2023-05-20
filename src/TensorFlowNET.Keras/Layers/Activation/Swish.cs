using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Swish : Layer {
            public Swish ( LayerArgs args ) : base(args) {
                  // Swish has no arguments
            }
            protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null) 
            {
                  Tensor x = inputs;

                  // x / (1 + exp(-x))
                  return tf.div(x, (tf.add(1f, tf.exp(tf.negative(x)))));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
