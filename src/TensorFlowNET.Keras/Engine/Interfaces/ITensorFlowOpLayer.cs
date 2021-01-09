using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine
{
    public interface ITensorFlowOpLayer
    {
        Layer GetOpLayer(TensorFlowOpLayerArgs args);
    }
}
