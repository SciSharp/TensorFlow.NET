using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Dot-product attention layer, a.k.a. Luong-style attention.
    /// Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
    /// shape `[batch_size, Tv, dim]` and `key` tensor of shape
    /// `[batch_size, Tv, dim]`. The calculation follows the steps:
    /// <para>
    /// 1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
    ///    product: `scores = tf.matmul(query, key, transpose_b=True)`.
    /// </para>
    /// <para>
    /// 2. Use scores to calculate a distribution with shape
    ///    `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
    /// </para>
    /// <para>
    /// 3. Use `distribution` to create a linear combination of `value` with
    ///    shape `[batch_size, Tq, dim]`:
    ///    `return tf.matmul(distribution, value)`.
    /// </para>
    /// </summary>
    /// <example> 0
    /// <code>
    /// //Variable-length int sequences.
    /// var query_input = keras.Input((1000), dtype: TF_DataType.TF_INT32);
    /// var value_input = keras.Input((1000), dtype: TF_DataType.TF_INT32);
    /// // Embedding lookup.
    /// var token_embedding = keras.layers.Embedding(input_dim: 1000, output_dim: 64);
    /// // Query embeddings of shape [batch_size, Tq, dimension].
    /// var query_embeddings = token_embedding.Apply(query_input);
    /// // Value embeddings of shape [batch_size, Tv, dimension].
    /// var value_embeddings = token_embedding.Apply(value_input);
    /// // CNN layer.
    /// var cnn_layer = keras.layers.Conv1D(
    ///     filters: 100,
    ///     kernel_size: 4,
    ///     // Use 'same' padding so outputs have the same shape as inputs.
    ///     padding: "same");
    /// var cnn_layer2 = keras.layers.Conv1D(
    ///     filters: 100,
    ///     kernel_size: 4,
    ///     // Use 'same' padding so outputs have the same shape as inputs.
    ///     padding: "same");
    /// // Query encoding of shape [batch_size, Tq, filters].
    /// var query_seq_encoding = cnn_layer.Apply(query_embeddings);
    /// // Value encoding of shape [batch_size, Tv, filters].
    /// var value_seq_encoding = cnn_layer.Apply(value_embeddings);
    /// // Query-value attention of shape [batch_size, Tq, filters].
    /// var query_value_attention_seq = keras.layers.Attention().Apply(
    ///    (query_seq_encoding, value_seq_encoding));
    /// // Reduce over the sequence axis to produce encodings of shape
    /// // [batch_size, filters].
    /// var query_encoding = keras.layers.GlobalAveragePooling1D().Apply(
    ///     query_seq_encoding);
    /// var query_value_attention = keras.layers.GlobalAveragePooling1D().Apply(
    ///     query_value_attention_seq);
    /// // Concatenate query and document encodings to produce a DNN input layer.
    /// var input_layer = keras.layers.Concatenate().Apply(
    ///     (query_encoding, query_value_attention));
    /// // Add DNN layers, and create Model.
    /// // ...
    /// </code>
    /// </example>
    public class Attention : BaseDenseAttention
    {
        
        public IVariableV1 concat_score_weight;
        
        public IVariableV1 scale;

        AttentionArgs args;
        
        string score_mode { get => args.score_mode; }
        
        bool use_scale { get => args.use_scale; }
        
        public Attention(AttentionArgs args) : base(args)
        {
            this.args = args;
            if (!new List<string> {
                "dot",
                "concat"
            }.Contains(this.score_mode))
                throw new ValueError("Received: score_mode={score_mode}. Acceptable values are: [\"dot\", \"concat\"]");
        }

        // Creates variable when `use_scale` is True or `score_mode` is `concat`.
        public override void build(KerasShapesWrapper input_shape)
        {
            if (this.use_scale)
                this.scale = this.add_weight(name: "scale",
                                             shape: 1,
                                             initializer: tf.ones_initializer,
                                             dtype: this.DType,
                                             trainable: true);
            else
                this.scale = null;

            if (this.score_mode == "concat")
                this.concat_score_weight = this.add_weight(name: "concat_score_weight",
                                                           shape: 1,
                                                           initializer: tf.ones_initializer,
                                                           dtype: this.DType,
                                                           trainable: true);
            else
                this.concat_score_weight = null;
            base.build(input_shape);
        }

        /// <summary>
        /// Calculates attention scores as a query-key dot product.
        /// </summary>
        /// <param name="query">query: Query tensor of shape `[batch_size, Tq, dim]`.</param>
        /// <param name="key">key: Key tensor of shape `[batch_size, Tv, dim]`.</param>
        /// <returns>Tensor of shape `[batch_size, Tq, Tv]`.</returns>
        public override Tensor _calculate_scores(Tensor query, Tensor key)
        {
            Tensor scores = null;
            if (this.score_mode == "dot")
            {
                //scores = tf.matmul(query, key, transpose_b: true);
                //scores = tf.matmul(tf.squeeze(query),tf.squeeze(key), transpose_b: true);
                scores = tf.linalg.einsum("bij,bkj->bik", (query, key));
                if (this.scale != null)
                    scores *= this.scale.AsTensor();
            } else if (this.score_mode == "concat") {
                // Reshape tensors to enable broadcasting.
                // Reshape into [batch_size, Tq, 1, dim].
                var q_reshaped = tf.expand_dims(query, axis: -2);
                // Reshape into [batch_size, 1, Tv, dim].
                var k_reshaped = tf.expand_dims(key, axis: -3);
                if (this.scale != null)
                    scores = this.concat_score_weight.AsTensor() *
                             tf.reduce_sum(tf.tanh(this.scale.AsTensor() * (q_reshaped + k_reshaped)), axis: -1);
                else
                    scores = this.concat_score_weight.AsTensor() *
                        tf.reduce_sum(tf.tanh(q_reshaped + k_reshaped), axis: -1);
            }
            return scores;
        }

        public override IKerasConfig get_config() => this.args;
        //var config = new Dictionary<object, object> {
        //    {
        //        "use_scale",
        //        this.use_scale},
        //    {
        //        "score_mode",
        //        this.score_mode}};
        //var base_config = base.get_config();
        //return new dict(base_config.items().ToList() + config.items().ToList());
    }
}
