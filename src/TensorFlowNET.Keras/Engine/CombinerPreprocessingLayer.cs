using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine
{
    public class CombinerPreprocessingLayer : Layer
    {
        PreprocessingLayerArgs args;

        public CombinerPreprocessingLayer(PreprocessingLayerArgs args)
            : base(args)
        {
            
        }
    }
}
