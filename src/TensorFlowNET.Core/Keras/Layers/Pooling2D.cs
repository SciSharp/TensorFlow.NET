using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    public class Pooling2D : Tensorflow.Layers.Layer
    {
        private Func<Tensor> pool_function;
        private int[] pool_size;
        private int[] strides;
        private string padding;
        private string data_format;
        private InputSpec input_spec;

        public Pooling2D(Func<Tensor> pool_function,
            int[] pool_size,
            int[] strides,
            string padding = "valid",
            string data_format = null,
            string name = null) : base(name: name)
        {
            this.pool_function = pool_function;
            this.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size");
            this.strides = conv_utils.normalize_tuple(strides, 2, "strides");
            this.padding = conv_utils.normalize_padding(padding);
            this.data_format = conv_utils.normalize_data_format(data_format);
            this.input_spec = new InputSpec(ndim: 4);
        }
    }
}
