using System.Collections.Generic;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class RNNArgs : LayerArgs
    {
        public interface IRnnArgCell : ILayer
        {
            object state_size { get; }
        }

        public IRnnArgCell Cell { get; set; } = null;
        public bool ReturnSequences { get; set; } = false;
        public bool ReturnState { get; set; } = false;
        public bool GoBackwards { get; set; } = false;
        public bool Stateful { get; set; } = false;
        public bool Unroll { get; set; } = false;
        public bool TimeMajor { get; set; } = false;
        public Dictionary<string, object> Kwargs { get; set; } = null;
    }
}
