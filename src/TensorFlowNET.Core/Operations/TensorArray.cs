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
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    /// <summary>
    /// TensorArray is designed to hide an underlying implementation object
    /// and as such accesses many of that object's hidden fields.
    ///
    /// "Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.
    /// This class is meant to be used with dynamic iteration primitives such as
    /// `while_loop` and `map_fn`.  It supports gradient back-propagation via special
    /// "flow" control flow dependencies.
    /// </summary>
    public class TensorArray
    {
        _GraphTensorArray _implementation;

        public TF_DataType dtype => _implementation._dtype;
        public Tensor handle => _implementation._handle;
        public Tensor flow => _implementation._flow;

        public TensorArray(TF_DataType dtype, Tensor size = default, bool? clear_after_read = null, bool? dynamic_size = null,
            string tensor_array_name = null, Tensor handle = null, Tensor flow = null,
            bool infer_shape = true, TensorShape element_shape = null,
            bool colocate_with_first_write_call = true, string name = null)
        {
            _implementation = new _GraphTensorArray(dtype, 
                size: size,
                dynamic_size: dynamic_size,
                clear_after_read: clear_after_read,
                tensor_array_name: tensor_array_name,
                handle: handle,
                flow: flow,
                infer_shape: infer_shape,
                element_shape: element_shape,
                colocate_with_first_write_call: colocate_with_first_write_call,
                name: name);
        }

        public TensorArray unstack(Tensor value, string name = null)
            => _implementation.unstack(value, name: name);
    }
}
