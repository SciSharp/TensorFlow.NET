using System;
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

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
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
