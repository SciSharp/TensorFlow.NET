﻿using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers {
      public class Cropping1D : Layer {
            CroppingArgs args;
            public Cropping1D ( CroppingArgs args ) : base(args) {
                  this.args = args;
            }

            protected override void build ( Tensors inputs ) {
                  if ( args.cropping.rank != 1 ) {
                        // throw an ValueError exception
                        throw new ValueError("");
                  }
                  else if ( args.cropping.shape[0] > 2 || args.cropping.shape[0] < 1 ) {
                        throw new ValueError("The `cropping` argument must be a tuple of 2 integers.");
                  }
                  built = true;
            }

            protected override Tensors Call ( Tensors inputs, Tensor state = null, bool? training = null ) {
                  Tensor output = inputs;
                  if ( output.rank != 3 ) {
                        // throw an ValueError exception
                        throw new ValueError("Expected dim=3, found dim=" + output.rank);
                  }
                  if ( args.cropping.shape[0] == 1 ) {
                        int crop_start = args.cropping[0];
                        output = output[new Slice(), new Slice(crop_start, ( int ) output.shape[1] - crop_start), new Slice()];
                  }
                  else {
                        int crop_start = args.cropping[0], crop_end = args.cropping[1];
                        output = output[new Slice(), new Slice(crop_start, ( int ) (output.shape[1]) - crop_end), new Slice()];
                  }
                  return output;
            }

            public override Shape ComputeOutputShape ( Shape input_shape ) {
                  if ( args.cropping.shape[0] == 1 ) {
                        int crop = args.cropping[0];
                        return new Shape(( int ) (input_shape[0]), ( int ) (input_shape[1] - crop * 2), ( int ) (input_shape[2]));
                  }
                  else {
                        int crop_start = args.cropping[0], crop_end = args.cropping[1];
                        return new Shape(( int ) (input_shape[0]), ( int ) (input_shape[1] - crop_start - crop_end), ( int ) (input_shape[2]));
                  }
            }
      }
}
