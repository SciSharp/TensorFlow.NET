using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
      public class Softmax : Layer {
            Axis axis;
            public Softmax ( SoftmaxArgs args ) : base(args) {
                  axis = args.axis;
            }
            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor x = inputs.Length == 2 ? inputs + ((1.0 - tf.cast(inputs[1], inputs.dtype)) * 1e-9)
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
