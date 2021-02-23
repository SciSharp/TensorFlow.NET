using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Sigmoid = (features, name) 
            => tf.Context.ExecuteOp("Sigmoid", name, new ExecuteOpArgs(features));
    }
}
