using System.Collections.Generic;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class SequentialArgs : ModelArgs
    {
        public List<Layer> Layers { get; set; }
    }
}
