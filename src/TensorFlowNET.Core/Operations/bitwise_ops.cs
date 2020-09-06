/*****************************************************************************
   Copyright 2020 Haiping Chen. All Rights Reserved.

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

using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class bitwise_ops
    {
        /// <summary>
        /// Elementwise computes the bitwise left-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/left_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor left_shift(Tensor x, Tensor y, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "LeftShift", name,
                    null,
                    x, y);

                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("LeftShift", name, args: new { x, y });
            return _op.output;
        }
    }
}
