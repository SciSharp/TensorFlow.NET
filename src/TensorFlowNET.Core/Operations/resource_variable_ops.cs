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
using Tensorflow.Framework;

namespace Tensorflow
{
    /// <summary>
    /// tensorflow\python\ops\resource_variable_ops.py
    /// </summary>
    public static class resource_variable_ops
    {
        public static ITensorOrOperation shape_safe_assign_variable_handle(Tensor handle, int[] shape, Tensor value, string name = null)
        {
            var value_tensor = ops.convert_to_tensor(value);
            return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name: name);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="self"></param>
        /// <param name="value"></param>
        /// <param name="use_locking"></param>
        /// <param name="read_value"></param>
        /// <returns>
        /// If `read_value` is `True`, this method will return the new value of the
        /// variable after the assignment has completed.Otherwise, when in graph mode
        /// it will return the `Operation` that does the assignment, and when in eager
        /// mode it will return `None`.
        /// </returns>
        public static Operation assign(this Tensor self, Tensor value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var value_tensor = ops.convert_to_tensor(value, dtype: self.dtype);
            self.assert_is_compatible_with(value_tensor);
            var assign_op = gen_resource_variable_ops.assign_variable_op(self, value_tensor, name: name);
            if (read_value)
            {
                return self._lazy_read(assign_op);
            }

            return assign_op;
        }

        public static Operation _lazy_read(this Tensor self, Operation op)
        {
            variable_accessed(self);
            throw new NotImplementedException();
        }

        public static void variable_accessed(this Tensor variable)
        {
            throw new NotImplementedException();
        }

        public static bool is_resource_variable(VariableV1 var)
        {
            return var is ResourceVariable;
        }

        /// <summary>
        /// Creates a variable handle with information to do shape inference.
        /// </summary>
        /// <param name="initial_value"></param>
        /// <param name="shape"></param>
        /// <param name="shared_name"></param>
        /// <param name="name"></param>
        /// <param name="graph_mode"></param>
        /// <returns></returns>
        public static Tensor eager_safe_variable_handle(Tensor initial_value, TensorShape shape, 
            string shared_name, string name, bool graph_mode)
        {
            var dtype = initial_value.dtype.as_base_dtype();
            return variable_handle_from_shape_and_dtype(
                shape, dtype, shared_name, name, graph_mode, initial_value);
        }

        public static Tensor variable_handle_from_shape_and_dtype(TensorShape shape, TF_DataType dtype, 
            string shared_name, string name, bool graph_mode, Tensor extra_handle_data = null)
        {
            throw new NotImplementedException("");
        }

        /// <summary>
        /// Represents a future for a read of a variable.
        /// Pretends to be the tensor if anyone looks.
        /// </summary>
        public class _UnreadVariable : BaseResourceVariable
        {
        }

        /// <summary>
        /// A python variable from an existing handle.
        /// </summary>
        public class BaseResourceVariable : VariableV1
        {
        }
    }
}
