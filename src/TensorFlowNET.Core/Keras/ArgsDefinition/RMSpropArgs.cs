using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class RMSpropArgs
    {
        public float LearningRate { get; set; } = 0.001f;
        public float RHO { get; set; } = 0.9f;
        public float Momentum { get; set; } = 0.0f;
        public float Epsilon { get; set; } = 1e-7f;
        public bool Centered { get; set; } = false;
        public string Name { get; set; } = "RMSprop";
    }
}
