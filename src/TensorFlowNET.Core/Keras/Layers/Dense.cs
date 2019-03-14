using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Dense : Tensorflow.Layers.Layer
    {
        protected int uints;
        protected IActivation activation;
        protected bool use_bias;
        protected IInitializer kernel_initializer;
        protected IInitializer bias_initializer;

        public Dense(int units,
            IActivation activation,
            bool use_bias = true,
            bool trainable = false,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null) : base(trainable: trainable)
        {
            this.uints = units;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            this.supports_masking = true;
            this.input_spec = new InputSpec(min_ndim: 2);
        }
    }
}
