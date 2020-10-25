using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class OptimizerV2Args
    {
        public string Name { get; set; }
        public float LearningRate { get; set; } = 0.001f;
        public float InitialDecay { get; set; }
        public float ClipNorm { get; set; }
        public float ClipValue { get; set; }
    }
}
