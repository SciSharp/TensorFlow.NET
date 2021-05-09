/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class string_ops
    {
        public Tensor lower(Tensor input, string encoding = "", string name = null)
            => tf.Context.ExecuteOp("StringLower", name, new ExecuteOpArgs(input, encoding));

        public Tensor regex_replace(Tensor input, string pattern, string rewrite,
                bool replace_global = true, string name = null)
                    => tf.Context.ExecuteOp("StaticRegexReplace", name, new ExecuteOpArgs(input)
                        .SetAttributes(new { pattern, rewrite, replace_global }));
        
        /// <summary>
        /// Return substrings from `Tensor` of strings.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="pos"></param>
        /// <param name="len"></param>
        /// <param name="name"></param>
        /// <param name="uint"></param>
        /// <returns></returns>
        public Tensor substr<T>(T input, int pos, int len,
                string @uint = "BYTE", string name = null)
            => tf.Context.ExecuteOp("Substr", name, new ExecuteOpArgs(input, pos, len)
                .SetAttributes(new { unit = @uint }));

        /// <summary>
        /// Computes the length of each string given in the input tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <param name="unit"></param>
        /// <returns></returns>
        public Tensor string_length(Tensor input, string name = null, string unit = "BYTE")
            => tf.Context.ExecuteOp("StringLength", name, new ExecuteOpArgs(input)
            {
                GetGradientAttrs = op => new
                {
                    unit = op.get_attr<string>("unit")
                }
            }.SetAttributes(new { unit }));

        public Tensor string_format(Tensor[] inputs, string template = "%s", string placeholder = "%s", int summarize = 3, string name = null)
            => tf.Context.ExecuteOp("StringFormat", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { inputs },
                GetGradientAttrs = op => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    template = op.get_attr<string>("template"),
                    placeholder = op.get_attr<string>("placeholder"),
                    summarize = op.get_attr<int>("summarize")
                }
            }.SetAttributes(new { template, placeholder, summarize }));

        public RaggedTensor string_split_v2(Tensor input, string sep = " ", int maxsplit = -1, string name = null)
        {
            return tf_with(ops.name_scope(name, "StringSplit"), scope =>
            {
                var sep_tensor = ops.convert_to_tensor(sep, dtype: TF_DataType.TF_STRING);
                if(input.rank == 0)
                {
                    var parts = string_split_v2(array_ops.stack(new[] { input }),
                        sep: sep,
                        maxsplit: maxsplit,
                        name: name);
                    return parts;
                }
                
                var result = tf.Context.ExecuteOp("StringSplitV2", name,
                    new ExecuteOpArgs(input, sep)
                    {
                        GetGradientAttrs = op => new
                        {
                            maxsplit = op.get_attr<int>("maxsplit")
                        }
                    }.SetAttributes(new { maxsplit }));
                var (indices, values, shape) = (result[0], result[1], result[2]);
                indices.set_shape(new TensorShape(-1, 2));
                values.set_shape(new TensorShape(-1));
                shape.set_shape(new TensorShape(2));

                var sparse_result = new SparseTensor(indices, values, shape);
                return RaggedTensor.from_value_rowids(sparse_result.values,
                    value_rowids: sparse_result.indices[Slice.All, 0],
                    nrows: sparse_result.dense_shape[0],
                    validate: false);
            });
        }

        public (RaggedTensor, RaggedTensor) unicode_decode_with_offsets(Tensor input, string input_encoding, string errors,
            int replacement_char = 0xFFFD, bool replace_control_characters = false, string name = null)
        {
            return tf_with(ops.name_scope(name, "UnicodeDecodeWithOffsets"), scope =>
            {
                var (codepoints, byte_start_offsets) = _unicode_decode(input, input_encoding, errors, 
                    replacement_char, replace_control_characters,
                    with_offsets: true, name: name);
                return (codepoints, byte_start_offsets);
            });
        }

        (RaggedTensor, RaggedTensor) _unicode_decode(Tensor input, string input_encoding, string errors, int replacement_char,
                    bool replace_control_characters, bool with_offsets, string name = null)
        {
            if (with_offsets)
            {
                var flat_result = tf.Context.ExecuteOp("UnicodeDecodeWithOffsets", name, new ExecuteOpArgs(input)
                {
                    GetGradientAttrs = op => new
                    {
                        input_encoding = op.get_attr<string>("input_encoding"),
                        errors = op.get_attr<string>("errors"),
                        replacement_char = op.get_attr<int>("replacement_char"),
                        replace_control_characters = op.get_attr<bool>("replace_control_characters"),
                        Tsplits = op.get_attr<TF_DataType>("Tsplits")
                    }
                }.SetAttributes(new
                {
                    input_encoding,
                    errors,
                    replacement_char,
                    replace_control_characters
                }));

                var codepoints = RaggedTensor.from_row_splits(flat_result[1], flat_result[0], validate: false);

                var offsets = RaggedTensor.from_row_splits(flat_result[2], flat_result[0], validate: false);
                return (codepoints, offsets);
            }

            return (null, null);
        }
    }
}
