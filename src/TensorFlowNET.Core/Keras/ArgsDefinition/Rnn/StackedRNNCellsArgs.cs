using System.Collections.Generic;
using Tensorflow.Keras.Layers.Rnn;

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class StackedRNNCellsArgs : LayerArgs
    {
        public IList<IRnnCell> Cells { get; set; }
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
