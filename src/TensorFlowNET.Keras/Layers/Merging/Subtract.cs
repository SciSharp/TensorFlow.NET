using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Subtract : Merge
    {
        public Subtract(MergeArgs args) : base(args)
        {

        }

        protected override Tensors _merge_function(Tensors inputs)
        {
            if (len(inputs) != 2)
                throw new ValueError($"A `Subtract` layer should be called on exactly 2 inputs");
            return inputs[0] - inputs[1];
        }
    }
}
