using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.ArgsDefinition.Core;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System;
using System.Linq;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    public class MultiHeadAttention : Layer
    {
        static readonly string _CHR_IDX = "abcdefghijklmnopqrstuvwxyz";

        MultiHeadAttentionArgs args;
        Shape _query_shape = null;
        Shape _key_shape = null;
        Shape _value_shape = null;
        bool _built_from_signature = false;
        EinsumDense _query_dense = null;
        EinsumDense _key_dense = null;
        EinsumDense _value_dense = null;
        EinsumDense _output_dense = null;
        string _dot_product_equation = "";
        string _combine_equation = "";
        Softmax _softmax = null;
        Dropout _dropout_layer = null;

        /// <summary>
        /// Builds einsum equations for the attention computation.
        /// Query, key, value inputs after projection are expected to have the shape as:
        /// `(bs, [non-attention dims], [attention dims], num_heads, channels)`.
        /// `bs` and `[non-attention dims]` are treated as `[batch dims]`.
        /// 
        /// <para>
        /// The attention operations can be generalized:
        /// </para>
        /// <para>
        ///   (1) Query-key dot product:
        ///   `([batch dims], [query attention dims], num_heads, channels), ([batch dims],
        ///   [key attention dims], num_heads, channels) -> ([batch dim],
        ///   num_heads, [query attention dims], [key attention dims])`
        ///   </para><para>
        ///   (2) Combination:
        ///   `([batch dims], num_heads, [query attention dims], [key attention dims]),
        ///   ([batch dims], [value attention dims], num_heads, channels) -> ([batch dims],
        ///   [query attention dims], num_heads, channels)`
        /// </para>
        /// </summary>
        /// <param name="rank">Rank of query, key, value tensors.</param>
        /// <param name="attn_axes">List/tuple of axes, `[-1, rank)`,
        ///                        that attention will be applied to.</param>
        /// <returns></returns>
        public static (string, string, int) _build_attention_equation(int rank, Shape attn_axes)
        {
            var target_notation = _CHR_IDX.Substring(0, rank);
            // `batch_dims` includes the head dim.
            // batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
            // Since range(rank) is an IEnumerable like (0, 1, 2 ...) whose index is equal to its value
            // use IEnumerable.Except instead of np.delete which is unavailable
            var batch_dims = range(rank).Except(attn_axes.as_int_list().concat(new[] { rank - 1 }));
            var letter_offset = rank;
            var source_notation = "";
            for (int i = 0; i < rank; i++)
            {
                if (batch_dims.Contains(i) || i == rank - 1)
                    source_notation += target_notation[i];
                else
                {
                    source_notation += _CHR_IDX[letter_offset];
                    letter_offset += 1;
                }
            }
            var product_notation = new string((from i in batch_dims
                                               select target_notation[i]).Concat(
                                               
                                               from i in attn_axes.as_int_list()
                                               select target_notation[i]).Concat(
                                               
                                               from i in attn_axes.as_int_list()
                                               select source_notation[i]).ToArray());
            var dot_product_equation = $"{source_notation},{target_notation}->{product_notation}";
            var attn_scores_rank = product_notation.Count();
            var combine_equation = $"{product_notation},{source_notation}->{target_notation}";
            return (dot_product_equation, combine_equation, attn_scores_rank);
        }

        /// <summary>
        /// Builds an einsum equation for projections inside multi-head attention.
        /// </summary>
        public static (string, string, int) _build_proj_equation(int free_dims, int bound_dims, int output_dims)
        {
            char _char;
            var input_str = "";
            var kernel_str = "";
            var output_str = "";
            var bias_axes = "";
            var letter_offset = 0;
            foreach (var i in range(free_dims))
            {
                _char = _CHR_IDX[i + letter_offset];
                input_str += _char;
                output_str += _char;
            }
            letter_offset += free_dims;
            foreach (var i in range(bound_dims))
            {
                _char = _CHR_IDX[i + letter_offset];
                input_str += _char;
                kernel_str += _char;
            }
            letter_offset += bound_dims;
            foreach (var i in range(output_dims))
            {
                _char = _CHR_IDX[i + letter_offset];
                kernel_str += _char;
                output_str += _char;
                bias_axes += _char;
            }
            var equation = $"{input_str},{kernel_str}->{output_str}";
            return (equation, bias_axes, output_str.Count());
        }

        static Shape _get_output_shape(int output_rank, Shape known_last_dims)
            => (from _ in range(output_rank - known_last_dims.rank)
                select -1).Concat(known_last_dims.as_int_list()).ToArray();

        public MultiHeadAttention(MultiHeadAttentionArgs args) : base(args)
        {
            this.args = args;
        }

        public void _build_from_signature(Tensor query, Tensor value, Tensor key = null)
            => this._build_from_signature(query.shape, value.shape, key?.shape);

        public void _build_from_signature(Shape query, Shape value, Shape key = null)
        {
            this._built_from_signature = true;
            this._query_shape = query;
            this._value_shape = value;
            if (key == null)
                this._key_shape = this._value_shape;
            else
                this._key_shape = key;
            // Any setup work performed only once should happen in an `init_scope`
            // to avoid creating symbolic Tensors that will later pollute any eager
            // operations.
            tf_with(tf.init_scope(), _ =>
            {
                var free_dims = this._query_shape.rank - 1;
                var (einsum_equation, bias_axes, output_rank) = _build_proj_equation(
                    free_dims, bound_dims: 1, output_dims: 2);
                this._query_dense = _get_dense(einsum_equation,
                                               _get_output_shape(output_rank - 1,
                                                                (this.args.NumHeads, this.args.KeyDim)),
                                               this.args.UseBias ? bias_axes : null,
                                               "query");
                (einsum_equation, bias_axes, output_rank) = _build_proj_equation(
                    this._key_shape.rank - 1, bound_dims: 1, output_dims: 2);
                this._key_dense = _get_dense(einsum_equation,
                                             _get_output_shape(output_rank - 1,
                                                              (this.args.NumHeads, this.args.KeyDim)),
                                             this.args.UseBias ? bias_axes : null,
                                             "key");
                (einsum_equation, bias_axes, output_rank) = _build_proj_equation(
                    this._value_shape.rank - 1, bound_dims: 1, output_dims: 2);
                this._value_dense = _get_dense(einsum_equation,
                                               _get_output_shape(output_rank - 1,
                                                                (this.args.NumHeads, this.args.ValueDim ?? this.args.KeyDim)),
                                               this.args.UseBias ? bias_axes : null,
                                               "value");
                // Builds the attention computations for multi-head dot product attention.
                // These computations could be wrapped into the keras attention layer once
                // it support mult-head einsum computations.
                this._build_attention(output_rank);
                this._output_dense = _build_output_dense(free_dims, "attention_output");
            });
            this.StackLayers(_query_dense, _key_dense, _value_dense, _output_dense);
        }

        EinsumDense _get_dense(string equation, Shape output_shape, string bias_axes, string name)
            => new EinsumDense(new EinsumDenseArgs()
            {
                Equation = equation,
                OutputShape = output_shape,
                BiasAxes = bias_axes,
                Name = name,
                KernelInitializer = this.args.KernelInitializer,
                BiasInitializer = this.args.BiasInitializer,
                KernelRegularizer = this.args.KernelRegularizer,
                BiasRegularizer = this.args.BiasRegularizer,
                KernelConstraint = this.args.KernelConstraint,
                BiasConstraint = this.args.BiasConstraint
            });

        EinsumDense _build_output_dense(int free_dims, string name)
        {
            if (this.args.OutputShape == null) this.args.OutputShape = new(this._query_shape[-1]);
            var (einsum_equation, bias_axes, output_rank) = _build_proj_equation(
                    free_dims, bound_dims: 2, output_dims: len(this.args.OutputShape));
            return _get_dense(einsum_equation,
                              _get_output_shape(output_rank - 1, this.args.OutputShape),
                              this.args.UseBias ? bias_axes : null,
                              name);
        }

        void _build_attention(int rank)
        {
            if (this.args.AttentionAxis == null)
                this.args.AttentionAxis = new(range(1, rank - 2).ToArray());
            int attn_scores_rank;
            (this._dot_product_equation, this._combine_equation, attn_scores_rank)
                = _build_attention_equation(rank, this.args.AttentionAxis);
            var norm_axes = range(attn_scores_rank - len(this.args.AttentionAxis),
                                  attn_scores_rank).ToArray();
            this._softmax = new Softmax(new SoftmaxArgs { axis = norm_axes });
            this._dropout_layer = new Dropout(new DropoutArgs { Rate = this.args.Dropout });
        }

        Tensor _masked_softmax(Tensor attention_scores, Tensor attention_mask = null)
        {
            if(attention_mask != null)
            {
                var mask_expansion_axis = -len(this.args.AttentionAxis) * 2 - 1;
                for (int i = 0; i < len(attention_scores.shape) - len(attention_mask.shape); i++)
                    attention_mask = tf.expand_dims(attention_mask, axis: mask_expansion_axis);
            }
            return this._softmax.Apply(attention_mask == null ? attention_scores : (attention_scores, attention_mask));
        }

        public Tensors _compute_attention(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor attention_mask = null,
            bool training = false)
        {
            // Note: Applying scalar multiply at the smaller end of einsum improves
            // XLA performance, but may introduce slight numeric differences in
            // the Transformer attention head.
            query = tf.multiply(query, 1f / tf.sqrt(tf.convert_to_tensor((float)this.args.KeyDim)));
            // Take the dot product between "query" and "key" to get the raw
            // attention scores.
            var attention_scores = tf.linalg.einsum(this._dot_product_equation, (key, query));
            attention_scores = this._masked_softmax(attention_scores, attention_mask);
            // This is actually dropping out entire tokens to attend to, which might
            // seem a bit unusual, but is taken from the original Transformer paper.
            var attention_scores_dropout = this._dropout_layer.Apply(attention_scores, training: training);
            // `context_layer` = [B, T, N, H]
            var attention_output = tf.linalg.einsum(this._combine_equation, (attention_scores_dropout, value));
            return (attention_output, attention_scores);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensors _inp;
            Tensor _mask = null;

            int count = inputs.Count();
            if (count < 2 || count > 5) throw new ValueError(
                    $"{ this.name } layer accepts inputs list of length from 2 to 5, " +
                    $"namely [query, value, (key), (attention_mask), (return_attention_scores)]." +
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
                    if (inputs[2].shape[-1] == inputs[1].shape[-1])
                        _inp = new[] { inputs[0], inputs[1], inputs[2] };
                    else
                    {
                        _inp = (inputs[0], inputs[1]);
                        _mask = inputs[2];
                    }
                    break;
                case 4:
                    _inp = new[] { inputs[0], inputs[1], inputs[2] };
                    _mask = inputs[3];
                    break;
                default:
                    throw new ValueError(); //TODO:Add discriptions for this err
            }

            return call(_inp, _mask, training, return_attention_scores);
        }

        protected Tensors call(Tensors inputs,
                               Tensor attention_mask,
                               bool? training = null,
                               bool return_attention_scores = false)
        {
            var (query, value, key) = (inputs[0], inputs[1], inputs.Length == 3 ? inputs[2] : null);
            if (!this._built_from_signature)
                this._build_from_signature(query: query, value: value, key: key);
            if (key == null)
                key = value;

            // TODO: Add RaggedTensor support
            //var query_is_ragged = query is tf.RaggedTensor;
            //if (query_is_ragged)
            //{
            //    var query_lengths = query.nested_row_lengths();
            //    query = query.to_tensor();
            //}
            //var key_is_ragged = key is tf.RaggedTensor;
            //var value_is_ragged = value is tf.RaggedTensor;
            //if (key_is_ragged && value_is_ragged)
            //{
            //    // Ensure they have the same shape.
            //    var bounding_shape = tf.math.maximum(key.bounding_shape(), value.bounding_shape());
            //    key = key.to_tensor(shape: bounding_shape);
            //    value = value.to_tensor(shape: bounding_shape);
            //}
            //else if (key_is_ragged)
            //{
            //    key = key.to_tensor(shape: tf.shape(value));
            //}
            //else if (value_is_ragged)
            //{
            //    value = value.to_tensor(shape: tf.shape(key));
            //}

            //   N = `num_attention_heads`
            //   H = `size_per_head`
            // `query` = [B, T, N ,H]
            query = this._query_dense.Apply(query);
            // `key` = [B, S, N, H]
            key = this._key_dense.Apply(key);
            // `value` = [B, S, N, H]
            value = this._value_dense.Apply(value);
            var (attention_output, attention_scores) = this._compute_attention(query, key, value, attention_mask, training ?? false);
            attention_output = this._output_dense.Apply(attention_output);

            //if (query_is_ragged)
            //{
            //    attention_output = tf.RaggedTensor.from_tensor(attention_output, lengths: query_lengths);
            //}

            if (return_attention_scores)
                return (attention_output, attention_scores.Single);
            return attention_output;
        }
    }
}