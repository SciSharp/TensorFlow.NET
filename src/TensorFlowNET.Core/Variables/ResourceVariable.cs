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

using Google.Protobuf;
using System;
using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Variable based on resource handles.
    /// </summary>
    public class ResourceVariable : VariableV1
    {
        bool _in_graph_mode;
        Tensor _handle;
        TensorShape _shape;
        public TensorShape shape => _shape;
        string _handle_name;
        string _unique_id;
        Operation _initializer_op;
        public override Operation initializer => _initializer_op;
        Tensor _initial_value;
        bool _trainable;
        public bool tranable => _trainable;
        Tensor _cached_value;
        Tensor _graph_element;
        public override Tensor graph_element => _graph_element;
        TF_DataType _dtype;
        public TF_DataType dtype => _dtype;
        public override string name => _handle.name;
        public string Device => _handle.Device;
        public Graph Graph => _handle.graph;
        public override Operation op => _handle.op;

        public ResourceVariable(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            VariableDef variable_def = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string import_scope = "",
            TensorShape shape = null) : base(initial_value,
                    trainable,
                    collections,
                    validate_shape,
                    caching_device,
                    name,
                    dtype)
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
            _graph_key = ops.get_default_graph().graph_key;

            ops.init_scope();
            _in_graph_mode = true;
            tf_with(ops.name_scope(name, "Variable"), scope =>
            {
                name = scope;
                var handle_name = ops.name_from_scope_name(name);
                var shared_name = handle_name;
                var unique_id = shared_name;

                var attr = new AttrValue();
                attr.List = new AttrValue.Types.ListValue();
                attr.List.S.Add(ByteString.CopyFromUtf8($"loc:{handle_name}"));
                tf_with(ops.name_scope("Initializer"), delegate
                {
                    initial_value = ops.convert_to_tensor(init_from_fn ? (initial_value as Func<Tensor>)() : initial_value,
                        name: "initial_value",
                        dtype: dtype);
                });
                _shape = shape ?? (initial_value as Tensor).TensorShape;
                _handle = resource_variable_ops.eager_safe_variable_handle(
                      initial_value: _initial_value,
                      shape: _shape,
                      shared_name: shared_name,
                      name: name,
                      graph_mode: _in_graph_mode);
                _unique_id = unique_id;
                _initial_value = initial_value as Tensor;
                _handle_name = handle_name + ":0";
                _dtype = _initial_value.dtype.as_base_dtype();
                // _constraint = constraint;

                if (_in_graph_mode)
                {
                    tf_with(ops.name_scope("IsInitialized"), delegate
                    {
                        _is_initialized_op = gen_resource_variable_ops.var_is_initialized_op(_handle);
                    });
                    if(initial_value != null)
                    {
                        tf_with(ops.name_scope("Assign"), scope1 =>
                        {
                            string n = scope1;
                            _initializer_op = gen_resource_variable_ops.assign_variable_op(_handle, 
                                variables._try_guard_against_uninitialized_dependencies(name, _initial_value),
                                name: n);
                        });
                    }
                }
            });

            throw new NotImplementedException("");
        }

        private void _init_from_proto(VariableDef variable_def, string import_scope = null)
        {
            _in_graph_mode = true;
            if (!variable_def.IsResource)
                throw new ValueError("Trying to restore Variable as ResourceVariable.");

            // Create from variable_def.
            var g = ops.get_default_graph();
            var prepend_name_scope = ops.prepend_name_scope(variable_def.VariableName, import_scope: import_scope);
            _handle = g.as_graph_element(prepend_name_scope) as Tensor;
            _shape = new TensorShape(_handle.op.get_attr("shape") as TensorShapeProto);
            _handle_name = _handle.name;
            _unique_id = _handle_name;
            prepend_name_scope = ops.prepend_name_scope(variable_def.InitializerName, import_scope: import_scope);
            _initializer_op = g.as_graph_element(prepend_name_scope) as Operation;
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

            _dtype = dtypes.as_tf_dtype((DataType)_handle.op.get_attr("dtype"));
        }

        public override string ToString()
        {
            return $"tf.ResourceVariable '{name}' shape={shape} dtype={dtype}";
        }
    }
}
