using System;
using Tensorflow.NumPy;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    public partial class LayersApi
    {
        public Attention Attention(bool use_scale = false,
                                   string score_mode = "dot",
                                   bool causal = false,
                                   float dropout = 0f) =>
            new Attention(new AttentionArgs
            {
                use_scale = use_scale,
                score_mode = score_mode,
                causal = causal,
                dropout = dropout
            });
    }
}