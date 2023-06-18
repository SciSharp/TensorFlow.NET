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
using Tensorflow.Common.Types;
using Tensorflow.Eager;
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

        Tensor _handle;
        public override Tensor handle => _handle;
        Tensor _flow;
        public override Tensor flow => _flow;
        bool _clear_after_read;
        List<Tensor> _tensor_array;
        List<int> _previous_read_indices;

        public _EagerTensorArray(TF_DataType dtype, Tensor size, bool dynamic_size = false,
            bool clear_after_read = true, string tensor_array_name = null, Tensor handle = null, Tensor flow = null,
            bool infer_shape = true, Shape? element_shape = null,
            bool colocate_with_first_write_call = true, string name = null)
        {
            _size = size;
            _flow = constant_op.constant(0);
            _infer_shape = infer_shape;
            _element_shape = element_shape ?? Shape.Null;
            _colocate_with_first_write_call = colocate_with_first_write_call;
            _dtype = dtype.as_base_dtype();
            _dynamic_size = dynamic_size;
            _clear_after_read = clear_after_read;
            _tensor_array = Enumerable.Repeat<Tensor>(null, size.numpy()).ToList();
            _previous_read_indices = new();
        }

        public override TensorArray unstack(Tensor value, string name = null)
        {
            var tensors = array_ops.unstack(value, name: name);
            if(tensors.Length > _tensor_array.Count && !_dynamic_size)
            {
                throw new ValueError($"Cannot unstack {tensors.Length} tensors into a TensorArray of static size {_tensor_array.Count}");
            }
            _tensor_array = tensors.ToList();
            // TODO(Rinne): revise the implementation. Here we should return `parent()`.
            return this;
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
            //if (indices is EagerTensor)
            //{
            //    indices = indices as EagerTensor;
            //    indices = indices.numpy();
            //}

            //foreach (var (index, val) in zip(indices.ToArray<int>(), array_ops.unstack(value)))
            //{
            //    this.write(index, val);
            //}
            //return base;
            //throw new NotImplementedException("");
            return this;
        }

        public void _merge_element_shape(Shape shape)
        {
            _element_shape.concatenate(shape);
        }

        public void _maybe_colocate_with(Tensor value)
        {
            _colocate_with.Add(value);
        }

        private Tensor _maybe_zero(int ix)
        {
            var val = _tensor_array[ix];
            if(val is null)
            {
                val = _tensor_array[ix] = array_ops.zeros(_element_shape, _dtype);
            }
            return val;
        }

        public override Tensor read<T>(T index, string name = null)
        {
            int index_int;
            if (index is int int_index)
                index_int = int_index;
            else if (index is Tensor tensor_index)
                index_int = tensor_index.numpy();
            else
                throw new ValueError("");

            if(index_int >= _tensor_array.Count)
            {
                throw new OutOfRangeError($"Tried to read from index {index_int} but array size is: {_tensor_array.Count} ");
            }

            var res = _tensor_array[index_int];
            if(res is null)
            {
                if (_previous_read_indices.Contains(index_int))
                {
                    throw new InvalidArgumentError($"Could not read index {index_int} twice because it was cleared after " +
                        $"a previous read (perhaps try setting clear_after_read = false?)");
                }
                else
                {
                    res = _maybe_zero(index_int);
                }
            }

            if (_clear_after_read)
            {
                _tensor_array[index_int] = null;
                _previous_read_indices.Add(index_int);
            }
            return res;
        }

        public override TensorArray write(Tensor index, Tensor value, string name = null)
        {
            int index_int;
            if(index is EagerTensor eager)
            {
                return write<Tensor>(eager.numpy(), value, name);
            }
            throw new InvalidArgumentError("The index is supposed to be an EagerTensor");
        }

        public override TensorArray write<T>(int index, T value, string name = null)
        {
            int size = _tensor_array.Count;
            if(index >= size)
            {
                if (!_dynamic_size)
                {
                    throw new OutOfRangeError($"Tried to write to index {index} but array is not resizeable and size " +
                        $"is: {size} ");
                }
                _tensor_array.AddRange(Enumerable.Repeat<Tensor>(null, index - size + 1));
            }

            Tensor tensor = ops.convert_to_tensor(value, preferred_dtype: _dtype, name: "value");
            
            if(_dtype != tensor.dtype)
            {
                throw new InvalidArgumentError($"TensorArray dtype is {_dtype.as_python_name()} but Op is " +
                    $"trying to write dtype {tensor.dtype.as_python_name()} ");
            }

            if (!_element_shape.is_compatible_with(tensor.shape))
            {
                throw new ValueError($"Incompatible shape for value ({tensor.shape}), expected ({_element_shape})");
            }

            if (_infer_shape)
            {
                _element_shape = _element_shape.merge_with(tensor.shape);
            }
            _tensor_array[index] = tensor;
            return this;
        }

        private Tensor size(string name = null)
        {
            return gen_data_flow_ops.tensor_array_size_v3(_handle, _flow, name: name);
        }

        public override Tensor stack(string name = null)
        {
            if(_tensor_array.Count > 0)
            {
                for(int i = 0; i < _tensor_array.Count; i++)
                {
                    _maybe_zero(i);
                }
            }
            if(_tensor_array.Count == 0 && _element_shape.IsFullyDefined)
            {
                return ops.convert_to_tensor(new Shape(new long[] { 0 }.Concat(_element_shape.dims).ToArray()), name: name, dtype: _dtype);
            }
            else
            {
                return ops.convert_to_tensor(_tensor_array, name: name, dtype: _dtype);
            }
            //ops.colocate_with(_handle);
            //return tf_with(ops.name_scope(name, "TensorArrayStack", new { _handle }), delegate
            //{
            //    return gather(math_ops.range(0, size()), name: name);
            //});
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
