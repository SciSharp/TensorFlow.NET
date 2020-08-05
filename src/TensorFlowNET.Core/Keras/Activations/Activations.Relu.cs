using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Relu = (features, name) =>
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Relu", name,
                    null,
                    features);

                return results[0];
            }

            throw new NotImplementedException("");
        };
    }
}
