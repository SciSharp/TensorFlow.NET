using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Operations;

namespace Tensorflow.Keras.Layers
{
    public class AveragePooling2D : Pooling2D
    {
        public AveragePooling2D(AveragePooling2DArgs args)
            : base(args)
        {
            args.PoolFunction = new AveragePoolFunction();
        }
    }
}
