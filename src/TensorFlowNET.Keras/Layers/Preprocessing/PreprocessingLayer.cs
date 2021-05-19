using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class PreprocessingLayer : Layer
    {
        public PreprocessingLayer(PreprocessingLayerArgs args) : base(args)
        {

        }
    }
}
