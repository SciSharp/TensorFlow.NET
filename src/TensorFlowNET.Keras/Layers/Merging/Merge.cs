using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public abstract class Merge : Layer
    {
        public Merge(MergeArgs args) : base(args)
        {

        }

        protected override void build(Tensors inputs)
        {
            // output_shape = input_shape.dims[1^];
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
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
