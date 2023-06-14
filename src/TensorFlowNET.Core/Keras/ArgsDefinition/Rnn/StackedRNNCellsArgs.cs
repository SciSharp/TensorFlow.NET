using System.Collections.Generic;
<<<<<<< HEAD
<<<<<<< HEAD
using Tensorflow.Keras.Layers.Rnn;
=======
using static Tensorflow.Keras.ArgsDefinition.Rnn.RNNArgs;
>>>>>>> master
=======
using Tensorflow.Keras.Layers.Rnn;
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class StackedRNNCellsArgs : LayerArgs
    {
<<<<<<< HEAD
<<<<<<< HEAD
        public IList<IRnnCell> Cells { get; set; }
=======
        public IList<IRnnArgCell> Cells { get; set; }
>>>>>>> master
=======
        public IList<IRnnCell> Cells { get; set; }
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
