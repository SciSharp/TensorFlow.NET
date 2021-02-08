using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers
{
    public class GlobalAveragePooling2D : GlobalPooling2D
    {
        public GlobalAveragePooling2D(Pooling2DArgs args)
            : base(args)
        {
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            if (data_format == "channels_last")
                return math_ops.reduce_mean(inputs, new int[] { 1, 2 }, false);
            else
                return math_ops.reduce_mean(inputs, new int[] { 2, 3 }, false);
        }
    }
}
