using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Upsampling layer for 2D inputs.
    /// </summary>
    public class UpSampling2D : Layer
    {
        UpSampling2DArgs args;
        int[] size;
        string data_format;
        string interpolation => args.Interpolation;

        public UpSampling2D(UpSampling2DArgs args) : base(args)
        {
            this.args = args;
            data_format = conv_utils.normalize_data_format(args.DataFormat);
            size = conv_utils.normalize_tuple(args.Size, 2, "size");
            inputSpec = new InputSpec(ndim: 4);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            return keras.backend.resize_images(inputs, 
                size[0], size[1], 
                data_format, 
                interpolation: interpolation);
        }
    }
}
