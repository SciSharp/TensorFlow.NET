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

namespace Tensorflow
{
    public partial class tensorflow
    {
        public Tensor scan(
            Func<Tensor, Tensor, Tensor> fn,
            Tensor elems,
            Tensor initializer = null,
            int parallel_iterations = 10,
            bool back_prop = true,
            bool swap_memory = false,
            bool infer_shape = true,
            bool reverse = false,
            string name = null) => functional_ops.scan(fn, elems, initializer, parallel_iterations, back_prop,
                swap_memory, infer_shape, reverse, name);
    }
}