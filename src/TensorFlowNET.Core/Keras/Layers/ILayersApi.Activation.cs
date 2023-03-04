using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.NumPy;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public ILayer ELU(float alpha = 0.1f);
        public ILayer SELU();
        public ILayer Softmax(int axis = -1);
        public ILayer Softmax(Axis axis);
        public ILayer Softplus();
        public ILayer HardSigmoid();
        public ILayer Softsign();
        public ILayer Swish();
        public ILayer Tanh();
        public ILayer Exponential();
    }
}
