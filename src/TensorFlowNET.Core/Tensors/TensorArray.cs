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

using Tensorflow.Common.Types;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
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
    public abstract class TensorArray : ITensorOrTensorArray
    {
        public virtual TF_DataType dtype { get; }
        public virtual Tensor handle { get; }
        public virtual Tensor flow { get; }
        public virtual bool infer_shape { get; }
        public virtual bool colocate_with_first_write_call { get; }

        public abstract TensorArray unstack(Tensor value, string name = null);

        public abstract Tensor read<T>(T index, string name = null);

        public abstract TensorArray write<T>(int index, T value, string name = null);
        public abstract TensorArray write(Tensor index, Tensor value, string name = null);

        public abstract Tensor stack(string name = null);
        public abstract Tensor gather(Tensor indices, string name = null);

        internal bool _dynamic_size;
        internal Tensor _size;
        internal List<Tensor> _colocate_with;
        internal Shape _element_shape;

        public static TensorArray Create(TF_DataType dtype, Tensor size = null, bool dynamic_size = false,
            bool clear_after_read = true, string tensor_array_name = null, Tensor handle = null, Tensor flow = null,
            bool infer_shape = true, Shape? element_shape = null,
            bool colocate_with_first_write_call = true, string name = null)
        {
            if (tf.Context.executing_eagerly() && (flow is null || flow.dtype != dtypes.variant))
            {
                return new _EagerTensorArray(dtype, size, dynamic_size, clear_after_read, tensor_array_name, handle, flow,
                    infer_shape, element_shape, colocate_with_first_write_call, name);
            }
            else
            {
                return new _GraphTensorArrayV2(dtype, size, dynamic_size, clear_after_read, tensor_array_name, handle, flow,
                    infer_shape, element_shape, colocate_with_first_write_call, name);
            }
        }
    }
}
