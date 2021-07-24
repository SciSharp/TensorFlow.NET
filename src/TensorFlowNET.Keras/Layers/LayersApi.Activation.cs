using Tensorflow.NumPy;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers {
      public partial class LayersApi {
            public ELU ELU ( float alpha = 0.1f )
                  => new ELU(new ELUArgs { Alpha = alpha });
            public SELU SELU ()
                  => new SELU(new LayerArgs { });
            public Softmax Softmax ( Axis axis ) => new Softmax(new SoftmaxArgs { axis = axis });
            public Softplus Softplus () => new Softplus(new LayerArgs { });
            public HardSigmoid HardSigmoid () => new HardSigmoid(new LayerArgs { });
            public Softsign Softsign () => new Softsign(new LayerArgs { });
            public Swish Swish () => new Swish(new LayerArgs { });
            public Tanh Tanh () => new Tanh(new LayerArgs { });
            public Exponential Exponential () => new Exponential(new LayerArgs { });
      }
}
