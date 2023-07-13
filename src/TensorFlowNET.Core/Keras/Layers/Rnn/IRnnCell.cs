using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public interface IRnnCell: ILayer
    {
        /// <summary>
        /// If the derived class tends to not implement it, please return null.
        /// </summary>
        INestStructure<long>? StateSize { get; }
        /// <summary>
        /// If the derived class tends to not implement it, please return null.
        /// </summary>
        INestStructure<long>? OutputSize { get; }
        /// <summary>
        /// Whether the optional RNN args are supported when appying the layer.
        /// In other words, whether `Apply` is overwrited with process of `RnnOptionalArgs`.
        /// </summary>
        bool SupportOptionalArgs { get; }
        Tensors GetInitialState(Tensors inputs, Tensor batch_size, TF_DataType dtype);
    }
}
