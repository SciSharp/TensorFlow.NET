﻿using Tensorflow.NumPy;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Common.Types;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Zero-padding layer for 2D input (e.g. picture).
    /// 
    /// This layer can add rows and columns of zeros
    /// at the top, bottom, left and right side of an image tensor.
    /// </summary>
    public class ZeroPadding2D : Layer
    {
        string data_format;
        NDArray padding;
        InputSpec input_spec;

        public ZeroPadding2D(ZeroPadding2DArgs args, string data_format = null)
            : base(args)
        {
            this.data_format = conv_utils.normalize_data_format(data_format);
            this.padding = args.Padding;
            this.input_spec = new InputSpec(ndim: 4);
        }

<<<<<<< HEAD
<<<<<<< HEAD
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
>>>>>>> master
=======
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
        {
            return keras.backend.spatial_2d_padding(inputs,
                padding: padding,
                data_format: data_format);
        }
    }
}
