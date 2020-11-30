using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    public abstract class GlobalPooling2D : Layer
    {
        Pooling2DArgs args;
        protected string data_format => args.DataFormat;
        protected InputSpec input_spec;

        public GlobalPooling2D(Pooling2DArgs args) : base(args)
        {
            this.args = args;
            args.DataFormat = conv_utils.normalize_data_format(data_format);
            input_spec = new InputSpec(ndim: 4);
        }
    }
}
