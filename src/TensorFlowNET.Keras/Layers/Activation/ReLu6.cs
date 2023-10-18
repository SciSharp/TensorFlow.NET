using System;
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
    public class ReLu6 : Layer
    {
        public ReLu6() : base(new LayerArgs { })
        {
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            return tf.nn.relu6(inputs);
        }
    }
}
