using Tensorflow.Common.Types;
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

<<<<<<< HEAD
<<<<<<< HEAD
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
>>>>>>> master
=======
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
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
