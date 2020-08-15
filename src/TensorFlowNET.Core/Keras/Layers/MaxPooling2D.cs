using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class MaxPooling2D : Pooling2D
    {
        public MaxPooling2D(MaxPooling2DArgs args) 
            : base(args)
        {
            args.PoolFunction = tf.nn.max_pool_fn;
        }
    }
}
