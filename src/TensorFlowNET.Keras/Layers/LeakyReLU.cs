using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Leaky version of a Rectified Linear Unit.
    /// </summary>
    public class LeakyReLU : Layer
    {
        LeakyReLUArgs args;
        float alpha => args.Alpha;
        public LeakyReLU(LeakyReLUArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            return tf.nn.leaky_relu(inputs, alpha: alpha);
        }
    }
}
