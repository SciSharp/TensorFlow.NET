using System.Collections.Generic;
<<<<<<< HEAD
using Tensorflow.Keras.Layers.Rnn;
=======
using static Tensorflow.Keras.ArgsDefinition.Rnn.RNNArgs;
>>>>>>> master

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class StackedRNNCellsArgs : LayerArgs
    {
<<<<<<< HEAD
        public IList<IRnnCell> Cells { get; set; }
=======
        public IList<IRnnArgCell> Cells { get; set; }
>>>>>>> master
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
