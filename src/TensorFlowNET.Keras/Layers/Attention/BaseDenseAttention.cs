using Tensorflow.Keras.Engine;
using Tensorflow.Keras.ArgsDefinition;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{

    /// <summary>
    /// Base Attention class for Dense networks.
    /// This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
    /// Attention is formed by three tensors: Query, Key and Value.
    /// This class is suitable for Dense or CNN networks, and not for RNN networks.
    /// Implementations of attention mechanisms should inherit from this class, and
    /// reuse the `apply_attention_scores()` method.
    /// </summary>
    public class BaseDenseAttention : Layer
    {

        BaseDenseAttentionArgs args;

        bool causal { get => args.causal; }
        
        float dropout { get => args.dropout; }

        protected bool supports_masking;
        
        public BaseDenseAttention(BaseDenseAttentionArgs args) : base(args)
        {
            this.args = args;
            this.supports_masking = true;
        }

        /// <summary>
        /// Calculates attention scores.
        /// </summary>
        /// <param name="query">query: Query tensor of shape `[batch_size, Tq, dim]`.</param>
        /// <param name="key">key: Key tensor of shape `[batch_size, Tv, dim]`.</param>
        /// <returns>Tensor of shape `[batch_size, Tq, Tv]`.</returns>
        public virtual Tensor _calculate_scores(Tensor query, Tensor key) =>
            throw new NotImplementedException("");

        /// <summary>
        /// Applies attention scores to the given value tensor.
        /// To use this method in your attention layer, follow the steps:
        /// <para>
        ///     * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
        ///       `[batch_size, Tv]` to calculate the attention `scores`.
        /// </para>
        /// <para>
        ///     * Pass `scores` and `value` tensors to this method. The method applies
        ///       `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
        ///       returns `matmul(attention_distribution, value).
        /// </para>
        /// <para>
        ///     * Apply `query_mask` and return the result.
        /// </para>
        /// </summary>
        /// <param name="scores">Scores float tensor of shape `[batch_size, Tq, Tv]`.</param>
        /// <param name="value">Value tensor of shape `[batch_size, Tv, dim]`.</param>
        /// <param name="scores_mask">
        /// A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
        /// [batch_size, Tq, Tv]`. If given, scores at positions where
        /// `scores_mask==False` do not contribute to the result. It must contain
        /// at least one `True` value in each line along the last dimension.
        /// </param>
        /// <param name="training">
        /// Boolean indicating whether the layer should behave in
        /// training mode (adding dropout) or in inference mode (no dropout).
        /// </param>
        /// <returns>
        /// <para>
        /// Tensor of shape `[batch_size, Tq, dim]`.
        /// </para>
        /// <para>
        /// Attention scores after masking and softmax with shape
        /// [batch_size, Tq, Tv]`.
        /// </para>
        /// </returns>
        public (Tensor, Tensor) _apply_scores(Tensor scores,
                                                      Tensor value,
                                                      Tensor scores_mask = null,
                                                      bool? training = null)
        {
            if (scores_mask != null)
            {
                var padding_mask = tf.logical_not(scores_mask);
                // Bias so padding positions do not contribute to attention distribution.
                // Note 65504. is the max float16 value.
                if (scores.dtype == tf.float16)
                    scores -= 65504f * tf.cast(padding_mask, dtype: scores.dtype);
                else
                    scores -= 1000000000f * tf.cast(padding_mask, dtype: scores.dtype);
            }
            bool _training;
            training ??= false; // TODO: Delete this line when backend.learning_phase is available
            if (training == null)
                _training = keras.backend.learning_phase() ==
                                Tensorflow.Keras.GraphLearningPhase.train_mode ?
                                true : false;
            else _training = training.Value;
            var weights = tf.nn.softmax(scores);
            Func<Tensor> dropped_weights = () => tf.nn.dropout(weights, rate: this.dropout);
            weights = Tensorflow.Framework.smart_module.smart_cond(_training, dropped_weights, () => tf.identity(weights));
            //return (tf.matmul(weights, value), weights);
            return (tf.linalg.einsum("bij,bjk->bik", (weights, value)), weights);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensors _inp;
            Tensors _mask = null;

            int count = inputs.Count();
            if (count < 2 || count > 6) throw new ValueError(
                    $"{ this.name } layer accepts inputs list of length from 2 to 6, " +
                    $"namely [query, value, (key), (query_mask), (value_mask), (return_attention_scores)]." +
                    $"Received length: {count}.");

            bool has_bool = inputs[count - 1].dtype == TF_DataType.TF_BOOL;
            bool return_attention_scores = false;
            if (has_bool)
            {
                return_attention_scores = (bool)inputs[count - 1];
                count--;
            }

            switch (count)
            {
                case 2:
                    _inp = (inputs[0], inputs[1]);
                    break;
                case 3:
                    _inp = new[] { inputs[0], inputs[1], inputs[2] };
                    break;
                case 4:
                    if (inputs[0].shape == inputs[2].shape)
                        if (inputs[1].shape == inputs[3].shape)
                        {
                            _inp = new[] { inputs[0], inputs[1] };
                            _mask = new[] { inputs[2], inputs[3] };
                            break;
                        }
                    throw new ValueError(); //TODO:Add discriptions for this err
                case 5:
                    _inp = new[] { inputs[0], inputs[1], inputs[2] };
                    _mask = (inputs[3], inputs[4]);
                    break;
                default:
                    throw new ValueError(); //TODO:Add discriptions for this err
            }

            return call(_inp, _mask, training, return_attention_scores);
        }

        protected Tensors call(Tensors inputs, Tensors mask = null, bool? training = null, bool return_attention_scores = false)
        {
            Tensor causal_mask;
            //this._validate_call_args(inputs: inputs, mask: mask);
            var q = inputs[0];
            var v = inputs[1];
            var k = inputs.Count() > 2 ? inputs[2] : v;
            var q_mask = mask != null ? mask[0] : null;
            var v_mask = mask != null ? mask[1] : null;
            var scores = this._calculate_scores(query: q, key: k);
            if (v_mask != null)
                // Mask of shape [batch_size, 1, Tv].
                v_mask = tf.expand_dims(v_mask, axis: -2);
            if (this.causal)
            {
                // Creates a lower triangular mask, so position i cannot attend to
                // positions j>i. This prevents the flow of information from the future
                // into the past.
                var scores_shape = tf.shape(scores);
                // causal_mask_shape = [1, Tq, Tv].
                var causal_mask_shape = tf.concat(new List<Tensor> {
                    tf.ones_like(tf.slice(scores_shape, new[]{0}, new[]{-2})),
                    tf.concat(new[]{scores_shape[-2], scores_shape[-1]}, 0)
                }, axis: 0);
                var _causal_mask_shape = new Shape(causal_mask_shape.ToArray<int>());
                causal_mask = _lower_triangular_mask(_causal_mask_shape);
            }
            else
                causal_mask = null;
            var scores_mask = _merge_masks(v_mask, causal_mask);
            var (result, attention_scores) = this._apply_scores(scores: scores, value: v, scores_mask: scores_mask, training: training);
            if (q_mask != null)
            {
                // Mask of shape [batch_size, Tq, 1].
                q_mask = tf.expand_dims(q_mask, axis: -1);
                result *= tf.cast(q_mask, dtype: result.dtype);
            }
            if (return_attention_scores)
                return new Tensors(result, attention_scores);
            return result;
        }
        
        public Tensor compute_mask(Tensors inputs, Tensors mask = null)
        {
            this._validate_call_args(inputs: inputs, mask: mask);
            if (mask != null)
            {
                var q_mask = mask[0];
                if (q_mask == null)
                    return null;
                return tf.convert_to_tensor(q_mask);
            }
            return null;
        }

        //public Shape compute_output_shape(Shape input_shape) {
        //    // return_attention_scores argument of BaseDenseAttention.call method
        //    // is ignored. Output shape of attention_scores cannot be returned.
        //    return input_shape[0];
        //}

        /// <summary>
        /// Validates arguments of the call method.
        /// </summary>
        public void _validate_call_args(Tensors inputs, Tensors mask)
        {
            if (inputs.Count() < 2 || inputs.Count() > 3)
                throw new ValueError(
                    $"{this.name} layer accepts inputs list of length 2 or 3, " +
                    $"namely [query, value] or [query, value, key]. Received length: {len(inputs)}.");
            if (mask != null)
                if (mask.Count() < 2 || mask.Count() > inputs.Count())
                    throw new ValueError($"{this.name} layer mask must be a list of length 2, " +
                                $"namely [query_mask, value_mask]. Received length: {len(mask)}.");
        }

        public static Tensor _lower_triangular_mask(Shape shape)
        {
            var row_index = tf.cumsum(tf.ones(shape: shape, dtype: tf.int32), axis: -2);
            var col_index = tf.cumsum(tf.ones(shape: shape, dtype: tf.int32), axis: -1);
            return tf.greater_equal(row_index, col_index);
        }

        public static Tensor _merge_masks(Tensor x, Tensor y)
        {
            if (x == null)
                return y;
            if (y == null)
                return x;
            return tf.logical_and(x, y);
        }

        public override IKerasConfig get_config() => this.args;
    }
}
