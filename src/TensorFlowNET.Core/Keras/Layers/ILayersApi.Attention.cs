using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public ILayer Attention(bool use_scale = false,
                                   string score_mode = "dot",
                                   bool causal = false,
                                   float dropout = 0f);
        public ILayer MultiHeadAttention(int num_heads,
                                            int key_dim,
                                            int? value_dim = null,
                                            float dropout = 0f,
                                            bool use_bias = true,
                                            Shape output_shape = null,
                                            Shape attention_axes = null,
                                            IInitializer kernel_initializer = null,
                                            IInitializer bias_initializer = null,
                                            IRegularizer kernel_regularizer = null,
                                            IRegularizer bias_regularizer = null,
                                            IRegularizer activity_regularizer = null,
                                            Action kernel_constraint = null,
                                            Action bias_constraint = null);
    }
}
