using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public class GlobalMaxPooling1D : GlobalPooling1D
    {
        public GlobalMaxPooling1D(Pooling1DArgs args)
            : base(args)
        {
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            if (data_format == "channels_last")
                return math_ops.reduce_max(inputs, 1, false);
            else
                return math_ops.reduce_max(inputs, 2, false);
        }
    }
}
