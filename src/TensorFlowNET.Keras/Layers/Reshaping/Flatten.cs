using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Flatten : Layer
    {
        FlattenArgs args;
        InputSpec input_spec;
        bool _channels_first;

        public Flatten(FlattenArgs args)
            : base(args)
        {
            args.DataFormat = conv_utils.normalize_data_format(args.DataFormat);
            input_spec = new InputSpec(min_ndim: 1);
            _channels_first = args.DataFormat == "channels_first";
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            if (_channels_first)
            {
                throw new NotImplementedException("");
            }

            if (tf.executing_eagerly())
            {
                return array_ops.reshape(inputs, new[] { inputs.shape[0], -1 });
            }

            throw new NotImplementedException("");
        }
    }
}
