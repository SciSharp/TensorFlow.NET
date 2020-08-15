using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class EmbeddingArgs : LayerArgs
    {
        public int InputDim { get; set; }
        public int OutputDim { get; set; }
        public bool MaskZero { get; set; }
        public int InputLength { get; set; } = -1;
        public IInitializer EmbeddingsInitializer { get; set; }
    }
}
