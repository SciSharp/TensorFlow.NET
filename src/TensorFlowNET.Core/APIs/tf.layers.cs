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
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                bool trainable = true,
                string name = null)
            {
                if (strides == null)
                    strides = new int[] { 1, 1 };
                if (dilation_rate == null)
                    dilation_rate = new int[] { 1, 1 };
                if (bias_initializer == null)
                    bias_initializer = tf.zeros_initializer;

                var layer = new Conv2D(filters,
                    kernel_size: kernel_size,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    dilation_rate: dilation_rate,
                    activation: activation,
                    use_bias: use_bias,
                    kernel_initializer: kernel_initializer,
                    bias_initializer: bias_initializer,
                    trainable: trainable,
                    name: name);

                return layer.apply(inputs);
            }
        }
    }
}
