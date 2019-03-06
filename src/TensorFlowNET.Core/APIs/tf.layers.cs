using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Activation;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class layers
        {
            public static Tensor conv2d(Tensor inputs,
                int filters,
                int[] kernel_size,
                int[] strides = null,
                string padding = "valid",
                string data_format= "channels_last",
                int[] dilation_rate = null,
                bool use_bias = true,
                IActivation activation = null,
                IInitializer kernel_initializer = null)
            {
                if (strides == null)
                    strides = new int[] { 1, 1 };
                if (dilation_rate == null)
                    dilation_rate = new int[] { 1, 1 };

                var layer = new Conv2D(filters, kernel_size);

                return layer.apply(inputs);
            }
        }
    }
}
