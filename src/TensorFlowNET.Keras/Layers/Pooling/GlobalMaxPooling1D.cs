using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers
{
    public class GlobalMaxPooling1D : GlobalPooling1D
    {
        public GlobalMaxPooling1D(Pooling1DArgs args)
            : base(args)
        {
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            if (data_format == "channels_last")
                return math_ops.reduce_max(inputs, new int[] { 1 }, false);
            else
                return math_ops.reduce_max(inputs, new int[] { 2 }, false);
        }
    }
}
