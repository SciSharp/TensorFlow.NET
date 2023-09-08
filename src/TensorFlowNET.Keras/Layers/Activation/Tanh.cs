using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Tanh : Layer
    {
        public Tanh(LayerArgs args) : base(args)
        {
            // Tanh has no arguments
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensor x = inputs;

            return tf.tanh(x);
        }
        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }
    }
}
