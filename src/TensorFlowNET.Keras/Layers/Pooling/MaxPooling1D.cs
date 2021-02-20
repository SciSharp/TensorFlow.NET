using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Operations;

namespace Tensorflow.Keras.Layers
{
    public class MaxPooling1D : Pooling1D
    {
        public MaxPooling1D(Pooling1DArgs args)
            : base(args)
        {
            args.PoolFunction = new MaxPoolFunction();
        }
    }
}
