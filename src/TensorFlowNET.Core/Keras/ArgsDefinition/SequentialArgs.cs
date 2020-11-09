using System.Collections.Generic;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class SequentialArgs : ModelArgs
    {
        public List<ILayer> Layers { get; set; }
    }
}
