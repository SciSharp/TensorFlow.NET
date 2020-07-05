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

using Google.Protobuf;
using NumSharp;
using System;
using System.Collections.Generic;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Variable based on resource handles.
    /// </summary>
    public partial class ResourceVariable : BaseResourceVariable
    {
        Tensor _cached_value;
        public string Device => handle.Device;
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        public Graph Graph => handle.graph;
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        public Operation op => handle.op;
        public Tensor is_initialized_op { get; set; }

        public ResourceVariable(IntPtr handle, IntPtr tensor) : base(handle, tensor)
        {
        }

        public ResourceVariable(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            VariableDef variable_def = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string import_scope = "",
            TensorShape shape = null)
        {
            if (variable_def != null)
            {
                if (initial_value != null)
                    throw new ValueError("variable_def and initial_value are mutually exclusive.");
                _init_from_proto(variable_def, import_scope: import_scope);
            }
            else
            {
                _init_from_args(initial_value: initial_value, 
                    trainable: trainable, 
                    collections: collections, 
                    caching_device: caching_device, 
                    name: name, 
                    dtype: dtype,
                    shape: shape);
            }

            // handle.ResourceVar = this;
        }

        private void _init_from_args(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            string caching_device = "",
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            TensorShape shape = null)
        {
            var init_from_fn = initial_value.GetType().Name == "Func`1";
            if(collections == null)
                collections = new List<string>() { tf.GraphKeys.GLOBAL_VARIABLES };
            _trainable = trainable;

            if (trainable && !collections.Contains(tf.GraphKeys.TRAINABLE_VARIABLES))
                collections.Add(tf.GraphKeys.TRAINABLE_VARIABLES);

            ops.init_scope();
            _in_graph_mode = !tf.context.executing_eagerly();
            tf_with(ops.name_scope(name, "Variable"), scope =>
            {
                name = scope;
                var handle_name = ops.name_from_scope_name(name);
                string unique_id = "";
                string shared_name = "";

                if (_in_graph_mode)
                {
                    shared_name = handle_name;
                    unique_id = shared_name;
                }
                else
                {
                    unique_id = $"{handle_name}_{ops.uid()}";
                    shared_name = tf.context.shared_name();
                }

                var attr = new AttrValue();
                attr.List = new AttrValue.Types.ListValue();
                attr.List.S.Add(ByteString.CopyFromUtf8($"loc:@{handle_name}"));
                tf_with(ops.name_scope("Initializer"), delegate
                {
                    initial_value = ops.convert_to_tensor(init_from_fn ? (initial_value as Func<Tensor>)() : initial_value,
                        name: "initial_value",
                        dtype: dtype);
                });
                _shape = shape ?? (initial_value as Tensor).TensorShape;
                _initial_value = initial_value as Tensor;
                handle = resource_variable_ops.eager_safe_variable_handle(
                      initial_value: _initial_value,
                      shape: _shape,
                      shared_name: shared_name,
                      name: name,
                      graph_mode: _in_graph_mode);
                
                _dtype = _initial_value.dtype.as_base_dtype();

                if (_in_graph_mode)
                {
                    tf_with(ops.name_scope("IsInitialized"), delegate
                    {
                        is_initialized_op = gen_resource_variable_ops.var_is_initialized_op(handle);
                    });

                    if(initial_value != null)
                    {
                        tf_with(ops.name_scope("Assign"), scope1 =>
                        {
                            string n = scope1;
                            initializer_op = gen_resource_variable_ops.assign_variable_op(handle, 
                                variables._try_guard_against_uninitialized_dependencies(name, _initial_value),
                                name: n);
                        });
                    }

                    // Manually assign reads to the handle's device to avoid log
                    // messages.
                    tf_with(ops.name_scope("Read"), delegate
                    {
                        var value = _read_variable_op();
                        _graph_element = value;
                    });

                    ops.add_to_collections(collections, this);
                }
                else
                {
                    gen_resource_variable_ops.assign_variable_op(handle, _initial_value);
                    is_initialized_op = null;
                    initializer_op = null;
                    _graph_element = null;
                    initial_value = _in_graph_mode ? initial_value : null;
                }

                base.__init__(trainable: trainable,
                    handle: handle,
                    name: name,
                    unique_id: unique_id,
                    handle_name: handle_name);
            });
        }

        private void _init_from_proto(VariableDef variable_def, string import_scope = null)
        {
            _in_graph_mode = true;
            if (!variable_def.IsResource)
                throw new ValueError("Trying to restore Variable as ResourceVariable.");

            // Create from variable_def.
            var g = ops.get_default_graph();
            var prepend_name_scope = ops.prepend_name_scope(variable_def.VariableName, import_scope: import_scope);
            handle = g.as_graph_element(prepend_name_scope) as Tensor;
            _shape = new TensorShape(handle.op.get_attr("shape") as TensorShapeProto);
            
            prepend_name_scope = ops.prepend_name_scope(variable_def.InitializerName, import_scope: import_scope);
            initializer_op = g.as_graph_element(prepend_name_scope) as Operation;
            if (!string.IsNullOrEmpty(variable_def.InitialValueName))
            {
                prepend_name_scope = ops.prepend_name_scope(variable_def.InitialValueName, import_scope: import_scope);
                _initial_value = g.as_graph_element(prepend_name_scope) as Tensor;
            }

            _trainable = variable_def.Trainable;
            /*var (synchronization, aggregation, trainable) =
        variables.validate_synchronization_aggregation_trainable(
            variable_def.Synchronization,
            variable_def.Aggregation,
            variable_def.Trainable,
            variable_def.VariableName);*/
            if (!string.IsNullOrEmpty(variable_def.SnapshotName))
            {
                prepend_name_scope = ops.prepend_name_scope(variable_def.SnapshotName, import_scope: import_scope);
                var snapshot = g.as_graph_element(prepend_name_scope) as Tensor;
                if (snapshot.op.type != "ReadVariableOp")
                    _cached_value = snapshot;
                while (snapshot.op.type != "ReadVariableOp")
                    snapshot = snapshot.op.inputs[0];
                _graph_element = snapshot;
            }
            else
            {
                throw new NotImplementedException("SnapshotName _init_from_proto");
            }

            if (variable_def.SaveSliceInfoDef != null)
            {
                throw new NotImplementedException("SaveSliceInfoDef _init_from_proto");
            }

            _dtype = dtypes.as_tf_dtype((DataType)handle.op.get_attr("dtype"));
        }

        public Tensor sparse_read(Tensor indices, string name = "Gather")
        {
            return tf_with(ops.name_scope(name), scope =>
            {
                name = scope;
                var value = gen_resource_variable_ops.resource_gather(
                    handle, indices, dtype: _dtype, name: name);

                return array_ops.identity(value);
            });
        }

        public override string ToString()
        {
            return $"tf.Variable: '{Name}' shape={string.Join(",", shape)}, dtype={dtype.as_numpy_name()}, numpy={EagerTensor.GetFormattedString(dtype, numpy())}";
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            // delete
            // c_api.TFE_DeleteResourceVariable(handle);
        }
    }
}
