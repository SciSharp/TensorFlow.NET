using System.Collections.Generic;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class StackedRNNCellsArgs : LayerArgs
    {
        public bool ReverseStateOrder = false;
    }
}
