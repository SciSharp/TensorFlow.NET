using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class RescalingArgs : LayerArgs
    {
        public float Scale { get; set; }
        public float Offset { get; set; }
    }
}
