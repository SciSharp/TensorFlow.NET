using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        public Activation Relu = (features, name)
            => tf.Context.ExecuteOp("Relu", name, new ExecuteOpArgs(features));
    }
}
