using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Rnn
{
    public abstract class RnnBase: Layer
    {
        public RnnBase(LayerArgs args): base(args) { }
    }
}
