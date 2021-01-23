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
using System.Linq;
using Tensorflow.Framework;
using static Tensorflow.CppShapeInferenceResult.Types;

namespace Tensorflow
{
    /// <summary>
    /// tensorflow\python\ops\resource_variable_ops.py
    /// </summary>
    public static class resource_variable_ops
    {
        public static Operation shape_safe_assign_variable_handle(Tensor handle, int[] shape, Tensor value, string name = null)
        {
            var value_tensor = ops.convert_to_tensor(value);
            return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name: name);
        }

        public static bool is_resource_variable(IVariableV1 var)
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

        /// <summary>
        /// Create a new variable handle, optionally copying in `extra_handle_data`
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="shared_name"></param>
        /// <param name="name"></param>
        /// <param name="graph_mode"></param>
        /// <param name="initial_value"></param>
        /// <returns></returns>
        public static Tensor variable_handle_from_shape_and_dtype(TensorShape shape, TF_DataType dtype,
            string shared_name, string name, bool graph_mode, Tensor initial_value = null)
        {
            var container = ops.get_default_graph().Container;
            var handle = gen_resource_variable_ops.var_handle_op(shape: shape,
                dtype: dtype,
                shared_name: shared_name,
                name: name,
                container: container);

            if (initial_value == null)
                initial_value = handle;

            if (graph_mode)
            {
                var full_handle_data = _combine_handle_data(handle, initial_value);
                _set_handle_shapes_and_types(handle, full_handle_data, graph_mode);
                return handle;
            }
            else
            {
                // We do not want two distinct ResourceVariable objects for the same
                // underlying resource in the runtime.
                // When in eager mode, explicitly ensure so here. When in graph mode, it's
                // ensured by always generating different variable names.
                var exists = gen_resource_variable_ops.var_is_initialized_op(handle);

                // We create an assert Op instead of checking right away in order to be
                // compatible with ASYNC execution mode. Further, since not all devices
                // support string tensors, we encode the assertion string in the Op name
                /*gen_logging_ops.assert(gen_math_ops.logical_not(exists),
                    new[] { exists },
                    name: "EagerVariableNameReuse");*/

                var handle_data = new HandleData();
                handle_data.IsSet = true;
                handle_data.ShapeAndType.Add(new HandleShapeAndType
                {
                    Dtype = dtype.as_datatype_enum(),
                    Shape = shape.as_proto()
                });
                _set_handle_shapes_and_types(handle, handle_data, graph_mode);
                return handle;
            }
        }

        /// <summary>
        /// Sets the shape inference result HandleData on tensor.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="handle_data"></param>
        /// <param name="graph_mode"></param>
        private static void _set_handle_shapes_and_types(Tensor tensor, HandleData handle_data, bool graph_mode)
        {
            if (!graph_mode)
                return;

            var size = handle_data.ShapeAndType.Count;

            var shapes = new IntPtr[size];
            var types = new DataType[size];
            var ranks = new int[size];

            for (int i = 0; i < size; i++)
            {
                var shapeAndType = handle_data.ShapeAndType[i];
                types[i] = shapeAndType.Dtype;
                ranks[i] = shapeAndType.Shape.UnknownRank ? -1 : shapeAndType.Shape.Dim.Count;
                var dims = shapeAndType.Shape.Dim.Select(x => x.Size).ToArray();
            }
        }

        /// <summary>
        /// Concats HandleData from tensors `handle` and `initial_value`.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="initial_value"></param>
        /// <returns></returns>
        private static HandleData _combine_handle_data(Tensor handle, Tensor initial_value)
        {
            var variable_handle_data = get_eager_safe_handle_data(initial_value);

            if (initial_value.dtype != dtypes.variant)
                return variable_handle_data;

            throw new NotImplementedException("");
        }

        private static HandleData get_eager_safe_handle_data(Tensor handle)
        {
            if (handle == IntPtr.Zero)
            {
                var data = new HandleData();
                data.ShapeAndType.Add(new HandleShapeAndType
                {
                    Shape = handle.TensorShape.as_proto(),
                    Dtype = handle.dtype.as_datatype_enum()
                });
                return data;
            }
            else
            {
                return HandleData.Parser.ParseFrom(handle.BufferToArray());
            }
        }
    }
}
