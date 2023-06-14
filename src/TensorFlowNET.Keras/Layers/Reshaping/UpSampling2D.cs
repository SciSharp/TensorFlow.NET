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
            return keras.backend.resize_images(inputs, 
                size[0], size[1], 
                data_format, 
                interpolation: interpolation);
        }
    }
}
