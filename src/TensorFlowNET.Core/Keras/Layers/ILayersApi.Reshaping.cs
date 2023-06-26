using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public ILayer Reshape(Shape target_shape);
        public ILayer Reshape(object[] target_shape);

        public ILayer UpSampling1D(
            int size
            );

        public ILayer UpSampling2D(Shape size = null,
                string data_format = null,
                string interpolation = "nearest");

        public ILayer ZeroPadding2D(NDArray padding);
    }
}
