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
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    internal class _GraphTensorArray
    {
        internal TF_DataType _dtype;
        public TF_DataType dtype => _dtype;

        /// <summary>
        /// Used to keep track of what tensors the TensorArray should be
        /// colocated with.  We choose to colocate the TensorArray with the
        /// first tensor written to it.
        /// </summary>
        bool _colocate_with_first_write_call;
        public bool colocate_with_first_write_call => _colocate_with_first_write_call;

        bool _infer_shape;
        public bool infer_shape => _infer_shape;
        public bool _dynamic_size;
        public List<TensorShape> _element_shape;

        public List<Tensor> _colocate_with;

        internal Tensor _handle;
        public Tensor handle => _handle;
        internal Tensor _flow;

        public _GraphTensorArray(TF_DataType dtype, Tensor size, bool? dynamic_size = null,
            bool? clear_after_read = null, string tensor_array_name = null, Tensor handle = null, Tensor flow = null, 
            bool infer_shape = true, TensorShape element_shape = null, 
            bool colocate_with_first_write_call = true, string name = null)
        {
            clear_after_read = clear_after_read ?? true;
            dynamic_size = dynamic_size ?? false;
            _dynamic_size = dynamic_size.Value;
            _dtype = dtype;

            _colocate_with_first_write_call = colocate_with_first_write_call;
            if (colocate_with_first_write_call)
                _colocate_with = new List<Tensor>();

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
                _element_shape = new List<TensorShape> { element_shape };
            }

            tf_with(ops.name_scope(name, "TensorArray", new { handle, size, flow }), scope =>
            {
                if(handle != null)
                {
                    _handle = handle;
                    _flow = flow;
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
                        (_handle, _flow) = create();
                    }
                    else
                    {
                        (_handle, _flow) = create();
                    }
                }
            });
        }

        public TensorArray unstack(Tensor value, string name = null)
        {
            return tf_with(ops.name_scope(name, "TensorArrayUnstack", new { _handle, value }), delegate
            {
                var num_elements = array_ops.shape(value)[0];
                return scatter(indices: math_ops.range(0, num_elements), value: value, name: name);
            });
        }

        public TensorArray scatter(Tensor indices, Tensor value, string name = null)
        {
            return tf_with(ops.name_scope(name, "TensorArrayScatter", new { _handle, value, indices }), delegate
            {
                value = ops.convert_to_tensor(value, preferred_dtype: _dtype, name: "value");
                if (_infer_shape)
                {
                    var shape = new TensorShape(value.TensorShape.dims.Skip(1).ToArray());
                    _merge_element_shape(shape);
                }

                _maybe_colocate_with(value);
                var flow_out = gen_data_flow_ops.tensor_array_scatter_v3(
                    handle: _handle,
                    indices: indices,
                    value: value,
                    flow_in: _flow,
                    name: name);

                var ta = new TensorArray(_dtype,
                    infer_shape:_infer_shape,
                    element_shape: _element_shape[0],
                    dynamic_size: _dynamic_size,
                    handle: _handle,
                    flow: flow_out,
                    colocate_with_first_write_call: _colocate_with_first_write_call);
                

                return ta;
            });
        }

        public void _merge_element_shape(TensorShape shape)
        {
            _element_shape.Add(shape);
        }

        public void _maybe_colocate_with(Tensor value)
        {
            _colocate_with.Add(value);
        }

        public Tensor read(Tensor index, string name = null)
        {
            var value = gen_data_flow_ops.tensor_array_read_v3(
                handle: _handle,
                index: index,
                flow_in: _flow,
                dtype: _dtype,
                name: name);

            if (_element_shape != null)
                value.set_shape(_element_shape[0].dims);

            return value;
        }
    }
}
