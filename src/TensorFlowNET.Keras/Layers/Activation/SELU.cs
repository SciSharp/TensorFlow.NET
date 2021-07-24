using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      /// <summary>
      /// SELU Layer:
      /// similar to ELU, but has pre-defined alpha and scale
      /// </summary>
      public class SELU : Layer {
            protected const float alpha = 1.67326324f, scale = 1.05070098f;
            public SELU ( LayerArgs args ) : base(args) {
                  // SELU has no arguments
            }
            protected override void build ( Tensors inputs ) {
                  if ( alpha < 0f ) {
                        throw new ValueError("Alpha must be a number greater than 0.");
                  }
                  built = true;
            }
            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor output = inputs;
                  return tf.where(output > 0f, scale * output, scale * alpha * (tf.exp(output) - 1f));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
