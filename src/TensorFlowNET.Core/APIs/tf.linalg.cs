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

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor diag(Tensor diagonal, string name = null)
            => gen_array_ops.diag(diagonal, name: name);

        public static Tensor matmul(Tensor a, Tensor b) 
            => gen_math_ops.mat_mul(a, b);

        public static Tensor batch_matmul(Tensor x, Tensor y)
            => gen_math_ops.batch_mat_mul(x, y);
    }
}
