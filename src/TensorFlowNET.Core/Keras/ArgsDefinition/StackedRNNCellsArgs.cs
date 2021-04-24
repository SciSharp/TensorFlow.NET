using System.Collections.Generic;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class StackedRNNCellsArgs : LayerArgs
    {
        public IList<RnnCell> Cells { get; set; }
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
