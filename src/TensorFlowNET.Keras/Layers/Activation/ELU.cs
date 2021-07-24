using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      /// <summary>
      /// ELU Layer:
      /// x = 0 when x > 0, x = alpha( e^x-1 ) elsewhere
      /// </summary>
      public class ELU : Layer {
            ELUArgs args;
            float alpha => args.Alpha;
            public ELU ( ELUArgs args ) : base(args) {
                  this.args = args;
            }
            protected override void build ( Tensors inputs ) {
                  if ( alpha < 0f ) {
                        throw new ValueError("Alpha must be a number greater than 0.");
                  }
                  built = true;
            }
            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor output = inputs;
                  if ( alpha != 1f ) {
                        output = tf.where(output > 0f, output, alpha * (tf.exp(output) - 1f));
                  }
                  return output;
            }

            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  return input_shape;
            }
      }
}
