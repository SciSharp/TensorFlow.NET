using Newtonsoft.Json;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class MultiHeadAttentionArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("num_heads")]
        public int NumHeads { get; set; }
        [JsonProperty("key_dim")]
        public int KeyDim { get; set; }
        [JsonProperty("value_dim")]
        public int? ValueDim { get; set; } = null;
        [JsonProperty("dropout")]
        public float Dropout { get; set; } = 0f;
        [JsonProperty("use_bias")]
        public bool UseBias { get; set; } = true;
        [JsonProperty("output_shape")]
        public Shape OutputShape { get; set; } = null;
        [JsonProperty("attention_axes")]
        public Shape AttentionAxis { get; set; } = null;
        [JsonProperty("kernel_initializer")]
        public IInitializer KernelInitializer { get; set; } = tf.glorot_uniform_initializer;
        [JsonProperty("bias_initializer")]
        public IInitializer BiasInitializer { get; set; } = tf.zeros_initializer;
        [JsonProperty("kernel_regularizer")]
        public IRegularizer KernelRegularizer { get; set; } = null;
        [JsonProperty("bias_regularizer")]
        public IRegularizer BiasRegularizer { get; set; } = null;
        [JsonProperty("kernel_constraint")]
        public Action KernelConstraint { get; set; } = null;
        [JsonProperty("bias_constraint")]
        public Action BiasConstraint { get; set; } = null;
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }

        // TODO: Add `key_shape`, `value_shape`, `query_shape`.
    }
}