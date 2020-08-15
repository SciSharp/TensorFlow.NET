using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LSTMArgs : RNNArgs
    {
        public int Units { get; set; }
        public Activation Activation { get; set; }
        public Activation RecurrentActivation { get; set; }
        public IInitializer KernelInitializer { get; set; }
        public IInitializer RecurrentInitializer { get; set; }
        public IInitializer BiasInitializer { get; set; }
        public bool UnitForgetBias { get; set; }
        public float Dropout { get; set; }
        public float RecurrentDropout { get; set; }
        public int Implementation { get; set; }
        public bool ReturnSequences { get; set; }
        public bool ReturnState { get; set; }
        public bool GoBackwards { get; set; }
        public bool Stateful { get; set; }
        public bool TimeMajor { get; set; }
        public bool Unroll { get; set; }
    }
}
