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

        public SparseTensor string_split_v2(Tensor input, string sep = "", int maxsplit = -1, string name = null)
        {
            return tf_with(ops.name_scope(name, "StringSplit"), scope =>
            {
                var sep_tensor = ops.convert_to_tensor(sep, dtype: TF_DataType.TF_STRING);
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
                return new SparseTensor(indices, values, shape);
            });
        }
    }
}
