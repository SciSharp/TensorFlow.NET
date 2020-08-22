using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class SequentialArgs : ModelArgs
    {
        public List<Layer> Layers { get; set; }
    }
}
