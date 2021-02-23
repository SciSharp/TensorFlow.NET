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

using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    /// <summary>
    /// Operations for bitwise manipulation of integers.
    /// https://www.tensorflow.org/api_docs/python/tf/bitwise
    /// </summary>
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
        public Tensor left_shift(Tensor x, Tensor y, string name = null) => binary_op(x, y, "LeftShift", name);

        /// <summary>
        /// Elementwise computes the bitwise right-shift of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/right_shift
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor right_shift(Tensor x, Tensor y, string name = null) => binary_op(x, y, "RightShift", name);

        /// <summary>
        /// Elementwise computes the bitwise inversion of `x`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/invert
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor invert(Tensor x, string name = null) => unary_op(x, "Invert", name);

        /// <summary>
        /// Elementwise computes the bitwise AND of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_and
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor bitwise_and(Tensor x, Tensor y, string name = null) => binary_op(x, y, "BitwiseAnd", name);

        /// <summary>
        /// Elementwise computes the bitwise OR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_or
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor bitwise_or(Tensor x, Tensor y, string name = null) => binary_op(x, y, "BitwiseOr", name);

        /// <summary>
        /// Elementwise computes the bitwise XOR of `x` and `y`.
        /// https://www.tensorflow.org/api_docs/python/tf/bitwise/bitwise_xor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor bitwise_xor(Tensor x, Tensor y, string name = null) => binary_op(x, y, "BitwiseXor", name);


        #region Private helper methods

        /// <summary>
        /// Helper method to invoke unary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        Tensor unary_op(Tensor x, string opName, string name)
            => tf.Context.ExecuteOp(opName, name, new ExecuteOpArgs(x));

        /// <summary>
        /// Helper method to invoke binary operator with specified name.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="opName"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        Tensor binary_op(Tensor x, Tensor y, string opName, string name)
            => tf.Context.ExecuteOp(opName, name, new ExecuteOpArgs(x, y));
        #endregion
    }
}
