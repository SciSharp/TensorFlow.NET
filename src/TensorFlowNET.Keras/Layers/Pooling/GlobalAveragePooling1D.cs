using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public class GlobalAveragePooling1D : GlobalPooling1D
    {
        public GlobalAveragePooling1D(Pooling1DArgs args)
            : base(args)
        {
        }

<<<<<<< HEAD
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
>>>>>>> master
        {
            if (data_format == "channels_last")
                return math_ops.reduce_mean(inputs, 1, false);
            else
                return math_ops.reduce_mean(inputs, 2, false);
        }
    }
}
