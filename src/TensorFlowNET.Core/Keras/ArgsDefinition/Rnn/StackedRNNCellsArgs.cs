using System.Collections.Generic;
using Tensorflow.Keras.Layers.Rnn;

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class StackedRNNCellsArgs : LayerArgs
    {
        public bool ReverseStateOrder = false;
    }
}
