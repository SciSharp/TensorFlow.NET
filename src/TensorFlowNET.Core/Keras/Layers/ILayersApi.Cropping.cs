using System;
using Tensorflow.Keras.ArgsDefinition.Reshaping;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public ILayer Cropping1D(NDArray cropping);
        public ILayer Cropping2D(NDArray cropping, Cropping2DArgs.DataFormat data_format = Cropping2DArgs.DataFormat.channels_last);
        public ILayer Cropping3D(NDArray cropping, Cropping3DArgs.DataFormat data_format = Cropping3DArgs.DataFormat.channels_last);
    }
}
