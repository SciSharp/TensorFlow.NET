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
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    internal class _GraphTensorArray
    {
        TF_DataType _dtype;

        /// <summary>
        /// Used to keep track of what tensors the TensorArray should be
        /// colocated with.  We choose to colocate the TensorArray with the
        /// first tensor written to it.
        /// </summary>
        bool _colocate_with_first_write_call;

        bool _infer_shape;
        List<TensorShape> _element_shape;

        object _colocate_with;

        public _GraphTensorArray(TF_DataType dtype, Tensor size, bool? dynamic_size = null,
            bool? clear_after_read = null, string tensor_array_name = null, Tensor handle = null, Tensor flow = null, 
            bool infer_shape = true, TensorShape element_shape = null, 
            bool colocate_with_first_write_call = true, string name = null)
        {
            clear_after_read = clear_after_read ?? true;
            dynamic_size = dynamic_size ?? false;

            _dtype = dtype;

            _colocate_with_first_write_call = colocate_with_first_write_call;
            if (colocate_with_first_write_call)
                _colocate_with = new Tensor[0];

            // Record the current static shape for the array elements. The element
            // shape is defined either by `element_shape` or the shape of the tensor
            // of the first write. If `infer_shape` is true, all writes checks for
            // shape equality.
            if(element_shape == null)
            {
                _infer_shape = infer_shape;
                _element_shape = new List<TensorShape> { };
            }
            else
            {
                _infer_shape = true;
                _element_shape = new List<TensorShape> { };
            }

            tf_with(ops.name_scope(name, "", new { handle, size, flow }), scope =>
            {
                if(handle != null)
                {

                }
                else
                {
                    Func<(Tensor, Tensor)> create = () => gen_data_flow_ops.tensor_array_v3(size,
                        dtype: dtype,
                        element_shape: element_shape,
                        identical_element_shapes: infer_shape,
                        dynamic_size: dynamic_size.Value,
                        clear_after_read: clear_after_read.Value,
                        tensor_array_name: tensor_array_name,
                        name: scope);

                    // Construct the TensorArray with an empty device.  The first
                    // write into the TensorArray from a Tensor with a set device
                    // will retroactively set the device value of this op.
                    if (colocate_with_first_write_call)
                    {
                        ops.colocate_with(ignore_existing: true);
                        create();
                    }
                    else
                    {

                    }
                }
            });
        }
    }
}
