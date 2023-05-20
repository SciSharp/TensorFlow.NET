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

        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            if (data_format == "channels_last")
                return math_ops.reduce_mean(inputs, (1, 2), false);
            else
                return math_ops.reduce_mean(inputs, (2, 3), false);
        }
    }
}
