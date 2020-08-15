using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Dropout : Layer
    {
        DropoutArgs args;

        public Dropout(DropoutArgs args)
            : base(args)
        {
            this.args = args;
        }

        protected override Tensor call(Tensor inputs, bool is_training = false, Tensor state = null)
        {
            var output = tf_utils.smart_cond(is_training,
                () => tf.nn.dropout(inputs,
                        noise_shape: get_noise_shape(inputs),
                        seed: args.Seed,
                        rate: args.Rate),
                () => array_ops.identity(inputs));

            return output;
        }

        Tensor get_noise_shape(Tensor inputs)
        {
            if (args.NoiseShape == null)
                return null;

            return null;
        }
    }
}
