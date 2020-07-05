﻿/*****************************************************************************
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
using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_state_ops
    {
        /// <summary>
        /// Holds state in the form of a tensor that persists across steps.
        /// Outputs a ref to the tensor state so it may be read or modified.
        /// </summary>
        /// <param name="shape">The shape of the variable tensor.</param>
        /// <param name="dtype">The type of elements in the variable tensor.</param>
        /// <param name="name"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <returns></returns>
        public static Tensor variable_v2(int[] shape, TF_DataType dtype, string name = null, string container = "", string shared_name = "")
        {
            var _op = tf._op_def_lib._apply_op_helper("VariableV2", name: name, args: new { dtype, shape, container, shared_name });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");
            _attrs["container"] = _op.get_attr("container");
            _attrs["shared_name"] = _op.get_attr("shared_name");

            return _result[0];
        }

        /// <summary>
        /// Update 'ref' by assigning 'value' to it
        /// </summary>
        /// <param name="ref"></param>
        /// <param name="value"></param>
        /// <param name="validate_shape"></param>
        /// <param name="use_locking"></param>
        /// <param name="name"></param>
        public static Tensor assign(Tensor @ref, object value, 
            bool validate_shape = true, 
            bool use_locking = true,
            string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Assign", name: name, args: new { @ref, value, validate_shape, use_locking });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["T"] = _op.get_attr("T");
            _attrs["validate_shape"] = _op.get_attr("validate_shape");
            _attrs["use_locking"] = _op.get_attr("use_locking");

            return _result[0];
        }

        public static Tensor assign(RefVariable @ref, object value,
            bool validate_shape = true,
            bool use_locking = true,
            string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Assign", name: name, args: new { @ref, value, validate_shape, use_locking });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["T"] = _op.get_attr("T");
            _attrs["validate_shape"] = _op.get_attr("validate_shape");
            _attrs["use_locking"] = _op.get_attr("use_locking");

            return _result[0];
        }

        public static Tensor assign(ResourceVariable @ref, object value,
            bool validate_shape = true,
            bool use_locking = true,
            string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Assign", name: name, args: new { @ref, value, validate_shape, use_locking });

            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["T"] = _op.get_attr("T");
            _attrs["validate_shape"] = _op.get_attr("validate_shape");
            _attrs["use_locking"] = _op.get_attr("use_locking");

            return _result[0];
        }

        public static Tensor assign_sub(RefVariable @ref,
            Tensor value,
            bool use_locking = false,
            string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("AssignSub", name: name, args: new { @ref, value, use_locking });

            return _op.outputs[0];
        }


        //  Update 'ref' by adding 'value' to it.
        //  This operation outputs "ref" after the update is done.
        //  This makes it easier to chain operations that need to use the reset value.
        //  Args:
        //    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
        //      Should be from a `Variable` node.
        //    value: A `Tensor`. Must have the same type as `ref`.
        //      The value to be added to the variable.
        //    use_locking: An optional `bool`. Defaults to `False`.
        //      If True, the addition will be protected by a lock;
        //        otherwise the behavior is undefined, but may exhibit less contention.
        //      name: A name for the operation(optional).
        //  Returns:
        //    A mutable `Tensor`. Has the same type as `ref`.
        public static Tensor assign_add<T>(RefVariable @ref, T value, bool use_locking = false, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("AssignAdd", name: name, args: new { @ref, value, use_locking });
            return _op.outputs[0];
        }

        /// <summary>
        /// Adds sparse updates to a variable reference.
        /// </summary>
        /// <param name="ref"></param>
        /// <param name="indices"></param>
        /// <param name="updates"></param>
        /// <param name="use_locking"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor scatter_add(RefVariable @ref, Tensor indices, Tensor updates, bool use_locking = false, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ScatterAdd", name: name, args: new { @ref, indices, updates, use_locking });
            return _op.outputs[0];
        }

        public static Tensor is_variable_initialized(RefVariable @ref, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("IsVariableInitialized", name: name, args: new { @ref });
            return _op.output;
        }
    }
}
