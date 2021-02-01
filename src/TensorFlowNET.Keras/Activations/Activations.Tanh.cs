using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Tanh = (features, name) =>
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Tanh", name,
                    null,
                    features);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Tanh", name: name, args: new { x = features });

            return _op.output;
        };
    }
}
