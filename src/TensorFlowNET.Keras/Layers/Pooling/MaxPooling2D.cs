using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Operations;

namespace Tensorflow.Keras.Layers
{
    public class MaxPooling2D : Pooling2D
    {
        public MaxPooling2D(MaxPooling2DArgs args)
            : base(args)
        {
            args.PoolFunction = new MaxPoolFunction();
        }
    }
}
