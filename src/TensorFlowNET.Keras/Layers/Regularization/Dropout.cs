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

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            if (training == null)
                training = false;

            var output = tf_utils.smart_cond(training.Value,
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
