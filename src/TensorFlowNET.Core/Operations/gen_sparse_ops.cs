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

using System.Collections.Generic;

namespace Tensorflow
{
    public class gen_sparse_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        /// <summary>
        /// Converts a sparse representation into a dense tensor.
        /// </summary>
        /// <param name="sparse_indices"></param>
        /// <param name="output_shape"></param>
        /// <param name="sparse_values"></param>
        /// <param name="default_value"></param>
        /// <param name="validate_indices"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor sparse_to_dense<T>(Tensor sparse_indices, 
            int[] output_shape, 
            T sparse_values,
            T default_value,
            bool validate_indices = true,
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SparseToDense", name, args: new
            {
                sparse_indices,
                output_shape,
                sparse_values,
                default_value,
                validate_indices
            });

            return _op.output;
        }
    }
}
