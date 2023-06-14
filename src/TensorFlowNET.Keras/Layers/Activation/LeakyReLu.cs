﻿using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Leaky version of a Rectified Linear Unit.
    /// </summary>
    public class LeakyReLu : Layer
    {
        LeakyReLuArgs args;
        float alpha => args.Alpha;
        public LeakyReLu(LeakyReLuArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            return tf.nn.leaky_relu(inputs, alpha: alpha);
        }
    }
}
