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
        public ILayer Attention(bool use_scale = false,
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
                                                     Action bias_constraint = null) =>
            new MultiHeadAttention(new MultiHeadAttentionArgs
            {
                NumHeads = num_heads,
                KeyDim = key_dim,
                ValueDim = value_dim,
                Dropout = dropout,
                UseBias = use_bias,
                OutputShape = output_shape,
                AttentionAxis = attention_axes,
                KernelInitializer = kernel_initializer ?? tf.glorot_uniform_initializer,
                BiasInitializer = bias_initializer ?? tf.zeros_initializer,
                KernelRegularizer = kernel_regularizer,
                BiasRegularizer = bias_regularizer,
                ActivityRegularizer = activity_regularizer,
                KernelConstraint = kernel_constraint,
                BiasConstraint = bias_constraint,
            });
    }
}