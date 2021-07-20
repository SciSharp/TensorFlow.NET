using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Softmax = (features, name) 
            => tf.Context.ExecuteOp("Softmax", name, new ExecuteOpArgs(features));
    }
}
