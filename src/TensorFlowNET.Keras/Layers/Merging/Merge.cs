﻿using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public abstract class Merge : Layer
    {
        public Merge(MergeArgs args) : base(args)
        {

        }

        public override void build(KerasShapesWrapper input_shape)
        {
            // output_shape = input_shape.dims[1^];
            _buildInputShape = input_shape;
        }

<<<<<<< HEAD
<<<<<<< HEAD
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
>>>>>>> master
=======
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
        {
            return _merge_function(inputs);
        }

        protected virtual Tensors _merge_function(Tensors inputs)
        {
            var output = inputs[0];
            foreach (var i in range(1, inputs.Length))
                output += inputs[i];
            return output;
        }
    }
}
