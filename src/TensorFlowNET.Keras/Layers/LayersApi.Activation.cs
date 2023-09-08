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
                  => new SELU(new SELUArgs { });
            public ILayer Softmax(int axis = -1) => new Softmax(new SoftmaxArgs { axis = axis });
            public ILayer Softmax ( Axis axis ) => new Softmax(new SoftmaxArgs { axis = axis });
            public ILayer Softplus () => new Softplus(new SoftplusArgs { });
            public ILayer HardSigmoid () => new HardSigmoid(new HardSigmoidArgs { });
            public ILayer Softsign () => new Softsign(new SoftsignArgs { });
            public ILayer Swish () => new Swish(new SwishArgs { });
            public ILayer Tanh () => new Tanh(new TanhArgs { });
            public ILayer Exponential () => new Exponential(new ExponentialArgs { });
      }
}
