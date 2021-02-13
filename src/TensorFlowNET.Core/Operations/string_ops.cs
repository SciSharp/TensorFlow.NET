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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class string_ops
    {
        public Tensor lower(Tensor input, string encoding = "", string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "StringLower", name,
                    null,
                    input, encoding);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("StringLower", name: name, args: new
            {
                input,
                encoding
            });

            return _op.output;
        }

        public Tensor regex_replace(Tensor input, string pattern, string rewrite,
                bool replace_global = true, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "StaticRegexReplace", name,
                    null,
                    input, 
                    "pattern", pattern,
                    "rewrite", rewrite, 
                    "replace_global", replace_global);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("StaticRegexReplace", name: name, args: new
            {
                input,
                pattern,
                rewrite,
                replace_global
            });

            return _op.output;
        }

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
        {
            if (tf.Context.executing_eagerly())
            {
                var input_tensor = tf.constant(input);
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Substr", name,
                    null,
                    input, pos, len,
                    "unit", @uint);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Substr", name: name, args: new
            {
                input,
                pos,
                len,
                unit = @uint
            });

            return _op.output;
        }
    }
}
