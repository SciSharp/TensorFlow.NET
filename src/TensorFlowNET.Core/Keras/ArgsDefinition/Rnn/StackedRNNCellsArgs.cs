using System.Collections.Generic;
using static Tensorflow.Keras.ArgsDefinition.Rnn.RNNArgs;

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class StackedRNNCellsArgs : LayerArgs
    {
        public IList<IRnnArgCell> Cells { get; set; }
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
