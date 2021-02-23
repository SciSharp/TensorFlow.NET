using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Tanh = (features, name) 
            => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(features));
    }
}
