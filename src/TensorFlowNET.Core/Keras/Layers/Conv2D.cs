using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Conv2D : Conv
    {
        private int filters;
        private int[] kernel_size;
        private int[] strides;

        public Conv2D(int filters,
            int[] kernel_size,
            int[] strides = null,
            string padding = "valid",
            string data_format = "channels_last",
            int[] dilation_rate = null,
            IActivation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true,
            string name = null)
        {

        }

        public Tensor apply(Tensor inputs)
        {
            throw new NotImplementedException("apply");
        }
    }
}
