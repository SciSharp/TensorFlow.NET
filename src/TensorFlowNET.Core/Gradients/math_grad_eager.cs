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

using System;
using System.Linq;
using Tensorflow.Eager;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Gradients for operators defined in math_ops.py.
    /// </summary>
    [RegisterGradientEager("math_grad")]
    public class math_grad_eager
    {
        [RegisterGradientEager("Mul")]
        public static Tensor[] _MulGrad(EagerOperation op, IntPtr[] grads)
        {
            var x = op.InputHandles[0];
            var y = op.InputHandles[1];
            var grad = grads[0];

            if (op.SkipInputIndices.Contains(1) &&
                EagerTensor.GetRank(grad) == 0)
            {
                return new Tensor[]
                {
                    null,//gen_math_ops.mul(grad, math_ops.conj(y)),
                    null
                };
            }

            if (_ShapesFullySpecifiedAndEqual(x, y, grad))
            {
                return new Tensor[]
                {
                    math_ops.multiply(grad, y),
                    math_ops.multiply(grad, x)
                };
            }

            throw new NotImplementedException("");
        }

        public static bool _ShapesFullySpecifiedAndEqual(IntPtr x, IntPtr y, IntPtr grad)
        {
            var x_shape = EagerTensor.GetDims(x);
            var y_shape = EagerTensor.GetDims(y);

            var grad_shape = EagerTensor.GetDims(grad);
            return x_shape != null &&
                y_shape != null &&
                Enumerable.SequenceEqual(x_shape, y_shape) &&
                Enumerable.SequenceEqual(y_shape, grad_shape) &&
                !x_shape.Contains(-1);
        }
    }
}
