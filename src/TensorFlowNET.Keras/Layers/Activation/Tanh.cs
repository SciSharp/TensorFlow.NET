using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Tanh : Layer
    {
        public Tanh(LayerArgs args) : base(args)
        {
            // Tanh has no arguments
        }
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
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
