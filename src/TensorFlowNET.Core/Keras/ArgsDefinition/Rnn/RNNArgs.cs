using System.Collections.Generic;

namespace Tensorflow.Keras.ArgsDefinition.Rnn
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

        public int Units { get; set; }
        public Activation Activation { get; set; }
        public Activation RecurrentActivation { get; set; }
        public bool UseBias { get; set; } = true;
        public IInitializer KernelInitializer { get; set; }
        public IInitializer RecurrentInitializer { get; set; }
        public IInitializer BiasInitializer { get; set; }

        // kernel_regularizer=None,
        // recurrent_regularizer=None,
        // bias_regularizer=None,
        // activity_regularizer=None,
        // kernel_constraint=None,
        // recurrent_constraint=None,
        // bias_constraint=None,
        // dropout=0.,
        // recurrent_dropout=0.,
        // return_sequences=False,
        // return_state=False,
        // go_backwards=False,
        // stateful=False,
        // unroll=False,
        // **kwargs):
    }
}
