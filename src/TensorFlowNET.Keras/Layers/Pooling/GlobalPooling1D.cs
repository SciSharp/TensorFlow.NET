using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    public abstract class GlobalPooling1D : Layer
    {
        Pooling1DArgs args;
        protected string data_format => args.DataFormat;
        protected InputSpec input_spec;

        public GlobalPooling1D(Pooling1DArgs args) : base(args)
        {
            this.args = args;
            args.DataFormat = conv_utils.normalize_data_format(data_format);
            input_spec = new InputSpec(ndim: 3);
        }
    }
}
