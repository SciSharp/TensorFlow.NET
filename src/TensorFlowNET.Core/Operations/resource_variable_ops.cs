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
using Tensorflow.Train;
using Tensorflow.Training.Saving.SavedModel;
using Tensorflow.Variables;
using static Tensorflow.CppShapeInferenceResult.Types;
using static Tensorflow.Binding;
using Tensorflow.Operations;
using System.Buffers;
using Tensorflow.Eager;
using Tensorflow.Graphs;

namespace Tensorflow
{
    /// <summary>
    /// tensorflow\python\ops\resource_variable_ops.py
    /// </summary>
    public static class resource_variable_ops
    {
        public static Operation shape_safe_assign_variable_handle(Tensor handle, int[] shape, Tensor value, string name = null)
        {
            // TODO(Rinne): deal with `_handle_graph`.
            var value_tensor = ops.convert_to_tensor(value);
            return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name: name);
        }

        public static bool is_resource_variable(object var)
        {
            return var is BaseResourceVariable;
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
        public static Tensor eager_safe_variable_handle(Tensor initial_value, Shape shape,
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
        public static Tensor variable_handle_from_shape_and_dtype(Shape shape, TF_DataType dtype,
            string shared_name, string name, bool graph_mode, Tensor initial_value = null)
        {
            var container = ops.get_default_graph().Container;
            if(container is null)
            {
                container = "";
            }
            if (!graph_mode)
            {
                if(shared_name is not null)
                {
                    throw new Exception("Using an explicit shared_name is not allowed when executing eagerly.");
                }
                shared_name = tf.Context.anonymous_name();
            }
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
                var handle_data = handle_data_util.create_handle_data(shape, dtype);
                if (initial_value is not null && initial_value.dtype == dtypes.variant)
                {
                    var extra_handle_data = get_eager_safe_handle_data(initial_value);
                    if (extra_handle_data is not null && extra_handle_data.IsSet)
                    {
                        if (!handle_data.IsSet || handle_data.ShapeAndType.Count != 1)
                        {
                            throw new RuntimeError($"Expected VarHandleOp to return a length==1 shape_and_type, " +
                                $"but saw: '{handle_data}'");
                        }
                        handle_data.ShapeAndType.AddRange(extra_handle_data.ShapeAndType);
                    }
                }
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
        internal unsafe static void _set_handle_shapes_and_types(Tensor tensor, HandleData handle_data, bool graph_mode)
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

            //tensor.HandleData = handle_data;
            //if (!graph_mode)
            //    return;

            //var shapes = handle_data.ShapeAndType.Select(x => x.Shape);
            //var types = handle_data.ShapeAndType.Select(x => x.Dtype).ToArray();
            //var ranks = shapes.Select(s => s.UnknownRank ? -1 : s.Dim.Count).ToArray();
            //var converted_shapes = shapes.Select<TensorShapeProto, Memory<int>>(s =>
            //{
            //    if (!s.UnknownRank)
            //    {
            //        return s.Dim.Select(d => (int)d.Size).ToArray();
            //    }
            //    else
            //    {
            //        return Memory<int>.Empty;
            //    }
            //}).ToArray();

            //List<MemoryHandle> handles = new();
            //IntPtr[] shapes_with_ptr = new IntPtr[converted_shapes.Length];
            //foreach(var (i, m) in enumerate(converted_shapes))
            //{
            //    if(m.IsEmpty)
            //    {
            //        shapes_with_ptr[i] = IntPtr.Zero;
            //    }
            //    else
            //    {
            //        var handle = m.Pin();
            //        handles.Add(handle);
            //        shapes_with_ptr[i] = new IntPtr(handle.Pointer);
            //    }
            //}

            //Status status = new();
            //// TODO(Rinne): enable it.
            //c_api.TF_GraphSetOutputHandleShapesAndTypes(tensor.op.graph.c_graph, tensor._as_tf_output(), 
            //    shapes_with_ptr.Length, shapes_with_ptr, ranks, types, status);
            //handles = null;
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

        /// <summary>
        /// Copies an existing variable to a new graph, with no initializer.
        /// </summary>
        /// <param name="variable"></param>
        public static UninitializedVariable copy_to_graph_uninitialized(ResourceVariable variable)
        {
            var new_variable = new UninitializedVariable(
                trainable: variable.Trainable,
                shape: variable.shape,
                dtype: variable.dtype,
                name: variable.SharedName,
                aggregation: variable.Aggregation,
                extra_handle_data: null);
            new_variable._maybe_initialize_trackable();
            return new_variable;
        }

        /// <summary>
        /// Writes additional information of the variable into the SavedObject proto.
        /// </summary>
        /// <param name="resource_variable"></param>
        /// <param name="proto"></param>
        /// <param name="options"></param>
        /// <param name="enforcing_naming"></param>
        public static void write_object_proto_for_resource_variable(BaseResourceVariable resource_variable, SavedObject proto, SaveOptions options, bool enforcing_naming = true)
        {
            // lack of API: `proto.Variable.SetInParent()`.
            if(enforcing_naming && !resource_variable.Name.EndsWith(":0"))
            {
                throw new ValueError($"Cowardly refusing to save variable {resource_variable.Name} because of " +
                    $"unexpected suffix in the name (expected ':0') which won't be restored.");
            }
            if(proto.Variable is null)
            {
                proto.Variable = new SavedVariable();
            }
            proto.Variable.Name = meta_graph.op_name(resource_variable.Name);
            proto.Variable.Trainable = resource_variable.Trainable;
            proto.Variable.Dtype = resource_variable.dtype.as_datatype_enum();
            // TODO: lack of API `proto.Variable.Synchronization = resource_variable.synchronization.value`.
            proto.Variable.Aggregation = resource_variable.Aggregation;
            proto.Variable.Shape = resource_variable.shape.as_proto();

            if (options.experimental_variable_policy.save_variable_devices())
            {
                if (!string.IsNullOrEmpty(resource_variable.Device))
                {
                    proto.Variable.Device = resource_variable.Device;
                }
            }
        }

        public static void _maybe_set_handle_data(TF_DataType dtype, Tensor handle, Tensor tensor)
        {
            if(dtype == dtypes.variant)
            {
                var handle_data = get_eager_safe_handle_data(handle);
                if(handle_data.IsSet && handle_data.ShapeAndType.Count > 1)
                {
                    tensor.HandleData = new HandleData()
                    {
                        IsSet = true
                    };
                    tensor.HandleData.ShapeAndType.AddRange(handle_data.ShapeAndType.Skip(1));
                }
            }
        }

        public static HandleData get_eager_safe_handle_data(Tensor handle)
        {
            if (handle.Handle == null)
            {
                var data = new HandleData();
                data.ShapeAndType.Add(new HandleShapeAndType
                {
                    Shape = handle.shape.as_shape_proto(),
                    Dtype = handle.dtype.as_datatype_enum()
                });
                return data;
            }
            else
            {
                return HandleData.Parser.ParseFrom(handle.BufferToArray());
            }
            //if(handle is EagerTensor)
            //{
            //    return handle.HandleData;
            //}
            //else
            //{
            //    return handle_data_util.get_resource_handle_data(handle);
            //}
        }

        public static void variable_accessed(IVariableV1 variable)
        {
            if (ops.get_default_graph() is FuncGraph func_graph)
            {
                func_graph.watch_variable(variable);
            }
            if (variable.Trainable)
            {
                foreach (var tape in tf.GetTapeSet())
                    tape.VariableAccessed(variable);
            }
        }
    }
}
