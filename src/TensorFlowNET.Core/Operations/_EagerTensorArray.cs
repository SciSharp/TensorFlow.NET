/*****************************************************************************
   Copyright 2022 Haiping Chen. All Rights Reserved.

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
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class _EagerTensorArray : TensorArray
    {
        TF_DataType _dtype;
        public override TF_DataType dtype => _dtype;

        /// <summary>
        /// Used to keep track of what tensors the TensorArray should be
        /// colocated with.  We choose to colocate the TensorArray with the
        /// first tensor written to it.
        /// </summary>
        bool _colocate_with_first_write_call;
        public override bool colocate_with_first_write_call => _colocate_with_first_write_call;

        bool _infer_shape;
        public override bool infer_shape => _infer_shape;
        public bool _dynamic_size;
        public Shape _element_shape;

        public List<Tensor> _colocate_with;

        Tensor _handle;
        public override Tensor handle => _handle;
        Tensor _flow;
        public override Tensor flow => _flow;
        bool _clear_after_read;
        List<Tensor> _tensor_array;

        public _EagerTensorArray(TF_DataType dtype, Tensor size, bool dynamic_size = false,
            bool clear_after_read = true, string tensor_array_name = null, Tensor handle = null, Tensor flow = null,
            bool infer_shape = true, Shape? element_shape = null,
            bool colocate_with_first_write_call = true, string name = null)
        {
            _flow = constant_op.constant(0);
            _infer_shape = infer_shape;
            _element_shape = element_shape ?? Shape.Null;
            _colocate_with_first_write_call = colocate_with_first_write_call;
            _dtype = dtype.as_base_dtype();
            _dynamic_size = dynamic_size;
            _clear_after_read = clear_after_read;
            _tensor_array = new List<Tensor>();
        }

        public override TensorArray unstack(Tensor value, string name = null)
        {
            return tf_with(ops.name_scope(name, "TensorArrayUnstack", new { _handle, value }), delegate
            {
                var num_elements = array_ops.shape(value)[0];
                return scatter(indices: math_ops.range(0, num_elements), value: value, name: name);
            });
        }

        public TensorArray scatter(Tensor indices, Tensor value, string name = null)
        {
            /*return tf_with(ops.name_scope(name, "TensorArrayScatter", new { _handle, value, indices }), delegate
            {
                value = ops.convert_to_tensor(value, preferred_dtype: _dtype, name: "value");
                if (_infer_shape)
                {
                    var shape = new Shape(value.shape.dims.Skip(1).ToArray());
                    _merge_element_shape(shape);
                }

                _maybe_colocate_with(value);
                var flow_out = gen_data_flow_ops.tensor_array_scatter_v3(
                    handle: _handle,
                    indices: indices,
                    value: value,
                    flow_in: _flow,
                    name: name);

                var ta = new _EagerTensorArray(_dtype,
                    infer_shape: _infer_shape,
                    element_shape: _element_shape[0],
                    dynamic_size: _dynamic_size,
                    handle: _handle,
                    flow: flow_out,
                    colocate_with_first_write_call: _colocate_with_first_write_call);


                return ta;
            });*/
            throw new NotImplementedException("");
        }

        public void _merge_element_shape(Shape shape)
        {
            _element_shape.concatenate(shape);
        }

        public void _maybe_colocate_with(Tensor value)
        {
            _colocate_with.Add(value);
        }

        public override Tensor read<T>(T index, string name = null)
        {
            int index_int = -1;
            if (index is int int_index)
                index_int = int_index;
            else if (index is Tensor tensor_index)
                index_int = tensor_index.numpy();
            else
                throw new ValueError("");

            if (_clear_after_read)
            {
                _tensor_array[index_int] = null;
            }

            return _tensor_array[index_int];
        }

        public override TensorArray write(Tensor index, Tensor value, string name = null)
        {
            if (_infer_shape)
                _element_shape = _element_shape.merge_with(value.shape);
            _tensor_array.add(value);
            return this;
        }

        public override TensorArray write<T>(int index, T value, string name = null)
        {
            var value_tensor = ops.convert_to_tensor(value, preferred_dtype: _dtype, name: "value");
            var index_tensor = ops.convert_to_tensor(index, name: "index");
            return write(index_tensor, value_tensor, name: name);
        }

        private Tensor size(string name = null)
        {
            return gen_data_flow_ops.tensor_array_size_v3(_handle, _flow, name: name);
        }

        public override Tensor stack(string name = null)
        {
            ops.colocate_with(_handle);
            return tf_with(ops.name_scope(name, "TensorArrayStack", new { _handle }), delegate
            {
                return gather(math_ops.range(0, size()), name: name);
            });
        }

        public override Tensor gather(Tensor indices, string name = null)
        {
            var element_shape = Shape.Null;

            var value = gen_data_flow_ops.tensor_array_gather_v3(
                handle: _handle,
                indices: indices,
                flow_in: _flow,
                dtype: _dtype,
                name: name,
                element_shape: element_shape);

            //if (element_shape != null)
            //value.set_shape(-1, element_shape.dims);

            return value;
        }
    }
}
