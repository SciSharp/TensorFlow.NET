using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class GRUArgs : AutoSerializeLayerArgs
    {
        public int Units { get; set; }
        public Activation Activation { get; set; }
        public Activation RecurrentActivation { get; set; }
        public bool UseBias { get; set; } = true;
        public float Dropout { get; set; } = .0f;
        public float RecurrentDropout { get; set; } = .0f;
        public IInitializer KernelInitializer { get; set; }
        public IInitializer RecurrentInitializer { get; set; }
        public IInitializer BiasInitializer { get; set; }
        public bool ReturnSequences { get;set; }
        public bool ReturnState { get;set; }
        public bool GoBackwards { get;set; }
        public bool Stateful { get;set; }
        public bool Unroll { get;set; }
        public bool TimeMajor { get;set; }
        public bool ResetAfter { get;set; }
        public int Implementation { get; set; } = 2;

    }

}
