using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers.Rnn
{
    public interface IRnnCell: ILayer
    {
        GeneralizedTensorShape StateSize { get; }
        GeneralizedTensorShape OutputSize { get; }
        bool IsTFRnnCell { get; }
        /// <summary>
        /// Whether the optional RNN args are supported when appying the layer.
        /// In other words, whether `Apply` is overwrited with process of `RnnOptionalArgs`.
        /// </summary>
        bool SupportOptionalArgs { get; }
    }
}
