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
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class state_ops
    {
        /// <summary>
        /// Create a variable Operation.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <returns></returns>
        public static Tensor variable_op_v2(int[] shape,
            TF_DataType dtype,
            string name = "Variable",
            string container = "",
            string shared_name = "") => gen_state_ops.variable_v2(shape,
                dtype,
                name: name,
                container: container,
                shared_name: shared_name);

        public static Tensor assign<T>(T @ref, object value,
            bool validate_shape = true,
            bool use_locking = true,
            string name = null)
        {
            return gen_state_ops.assign(@ref,
                value,
                validate_shape: validate_shape,
                use_locking: use_locking,
                name: name);
        }

        public static Tensor assign(IVariableV1 @ref, object value,
            bool validate_shape = true,
            bool use_locking = true,
            string name = null)
        {
            if (@ref.dtype.is_ref_dtype())
                return gen_state_ops.assign(@ref,
                    value,
                    validate_shape: validate_shape,
                    use_locking: use_locking,
                    name: name);
            else
                return @ref.assign(value, name: name);
        }

        public static Tensor assign_sub(IVariableV1 @ref,
            Tensor value,
            bool use_locking = false,
            string name = null) => @ref.dtype.is_ref_dtype() ?
                gen_state_ops.assign_sub(@ref,
                    value,
                    use_locking: use_locking,
                    name: name) :
                @ref.assign_sub(value, name: name);

        //"""Update 'ref' by adding 'value' to it.
        //
        //  This operation outputs "ref" after the update is done.
        //  This makes it easier to chain operations that need to use the reset value.
        //
        //  Args:
        //    ref: A mutable `Tensor`. Must be one of the following types:
        //      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
        //      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        //      Should be from a `Variable` node.
        //    value: A `Tensor`. Must have the same type as `ref`.
        //      The value to be added to the variable.
        //    use_locking: An optional `bool`. Defaults to `False`.
        //      If True, the addition will be protected by a lock;
        //      otherwise the behavior is undefined, but may exhibit less contention.
        //    name: A name for the operation (optional).
        //
        //  Returns:
        //    Same as "ref".  Returned as a convenience for operations that want
        //    to use the new value after the variable has been updated.
        public static Tensor assign_add<T>(IVariableV1 @ref,
            T value,
            bool use_locking = false,
            string name = null)
        {
            if (tf.executing_eagerly())
                return @ref.assign_add(value, use_locking: use_locking, name: name);
            else
                return gen_state_ops.assign_add(@ref, value, use_locking: use_locking, name: name);
        }

        public static Tensor scatter_add(IVariableV1 @ref, Tensor indices, Tensor updates, bool use_locking = false, string name = null)
        {
            if (@ref.dtype.is_ref_dtype())
                return gen_state_ops.scatter_add(@ref, indices, updates, use_locking: use_locking, name: name);

            throw new NotImplementedException("scatter_add");
        }

        public static Tensor is_variable_initialized(RefVariable @ref, string name = null)
        {
            if (@ref.dtype.is_ref_dtype())
                return gen_state_ops.is_variable_initialized(@ref: @ref, name: name);
            throw new NotImplementedException("");
            //return @ref.is_initialized(name: name);
        }
    }
}
