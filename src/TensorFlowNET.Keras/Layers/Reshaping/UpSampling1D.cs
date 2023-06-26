using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;


namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Upsampling layer for 1D inputs.
    /// </summary>
    public class UpSampling1D : Layer
    {
        UpSampling1DArgs args;
        int size;

        public UpSampling1D(UpSampling1DArgs args) : base(args)
        {
            this.args = args;
            size = args.Size;
            inputSpec = new InputSpec(ndim: 3);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var output = keras.backend.repeat_elements(inputs, size, axis: 1);
            return output;
        }
    }
}
