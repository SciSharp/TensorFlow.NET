using Tensorflow.NumPy;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers {
      public partial class LayersApi {
            public ILayer ELU ( float alpha = 0.1f )
                  => new ELU(new ELUArgs { Alpha = alpha });
            public ILayer SELU ()
                  => new SELU(new LayerArgs { });
            public ILayer Softmax ( Axis axis ) => new Softmax(new SoftmaxArgs { axis = axis });
            public ILayer Softplus () => new Softplus(new LayerArgs { });
            public ILayer HardSigmoid () => new HardSigmoid(new LayerArgs { });
            public ILayer Softsign () => new Softsign(new LayerArgs { });
            public ILayer Swish () => new Swish(new LayerArgs { });
            public ILayer Tanh () => new Tanh(new LayerArgs { });
            public ILayer Exponential () => new Exponential(new LayerArgs { });
      }
}
