using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers {
      public class Permute : Layer {
            int[] dims, permute;
            public Permute ( PermuteArgs args ) : base(args) {
                  this.dims = args.dims;
            }
            protected override void build ( Tensors inputs ) {
                  var rank = inputs.rank;
                  if ( dims.Length != rank - 1 ) {
                        throw new ValueError("Dimensions must match.");
                  }
                  permute = new int[inputs.rank];
                  dims.CopyTo(permute, 1);
                  built = true;
            }
            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor outputs = inputs;
                  return tf.transpose(outputs, new Axis(permute));
            }
            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  Shape output_shape = new Shape(input_shape.dims);
                  for ( int i = 0; i < dims.Length; i += 1 ) {
                        var d = dims[i];
                        var target_dim = input_shape[d];
                        output_shape[i + 1] = target_dim;
                  }
                  return output_shape;
            }
      }
}
