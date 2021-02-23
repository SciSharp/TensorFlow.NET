using System;
using System.Linq;
using Tensorflow.Framework;
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
            this.args = args;
            args.DataFormat = conv_utils.normalize_data_format(args.DataFormat);
            input_spec = new InputSpec(min_ndim: 1);
            _channels_first = args.DataFormat == "channels_first";
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            if (_channels_first)
            {
                throw new NotImplementedException("");
            }

            if (tf.executing_eagerly())
            {
                return array_ops.reshape(inputs, new[] { inputs.shape[0], -1 });
            }
            else
            {
                var input_shape = inputs.shape;
                var rank = inputs.shape.rank;
                if (rank == 1)
                    return array_ops.expand_dims(inputs, axis: 1);
                var batch_dim = tensor_shape.dimension_value(input_shape[0]);
                if (batch_dim != -1)
                {
                    return array_ops.reshape(inputs, new[] { batch_dim, -1 });
                }

                var non_batch_dims = ((int[])input_shape).Skip(1).ToArray();
                var num = 1;
                if (non_batch_dims.Length > 0)
                {
                    for (var i = 0; i < non_batch_dims.Length; i++)
                    {
                        num *= non_batch_dims[i];
                    }
                }
                return array_ops.reshape(inputs, new[] { inputs.shape[0], num });
            }
        }
    }
}
