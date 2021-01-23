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
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    [Obsolete]
    public partial class RefVariable : IVariableV1, IProtoBuf<VariableDef, RefVariable>
    {
        protected string _name;
        public string UniqueId => _name;
        public Tensor GraphElement { get; }
        public Tensor _variable;
        public Tensor Handle => _variable;
        protected string _graph_key;
        public Graph Graph => _variable.graph;

        public Tensor _is_initialized_op { get; set; }

        protected TF_DataType _dtype;

        public bool _in_graph_mode = true;
        public Tensor _initial_value;
        public bool _trainable;

        public Tensor _snapshot;
        public bool _save_slice_info;

        private Operation _initializer_op;
        public Operation Initializer => _initializer_op;
        public Operation Op => _variable.op;

        public TF_DataType dtype => _variable.dtype;
        public TensorShape shape => tensor_util.to_shape(_variable.shape);
        public string Device => "";

        public string Name => _variable.name;

        public Tensor eval() => _variable;

        public RefVariable(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            VariableDef variable_def = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string import_scope = "") : base()
        {
            _in_graph_mode = true;

            if (initial_value is Operation op)
            {
                _init_from_op(op);
            }
            else if (variable_def != null)
            {
                if (initial_value != null)
                    throw new ValueError("variable_def and initial_value are mutually exclusive.");
                _init_from_proto(variable_def, import_scope: import_scope);
            }
            else
            {
                _init_from_args(initial_value, trainable, collections, validate_shape, caching_device, name, dtype);
            }
        }

        private void _init_from_op(Operation op)
        {
            var g = ops.get_default_graph();
            _initializer_op = op;
            _variable = op.output;
        }

        private void _init_from_proto(VariableDef variable_def, string import_scope = "")
        {
            var g = ops.get_default_graph();

            _variable = g.as_graph_element(
                ops.prepend_name_scope(variable_def.VariableName,
                                import_scope: import_scope)) as Tensor;

            _initializer_op = g.as_graph_element(
                ops.prepend_name_scope(variable_def.InitializerName,
                               import_scope: import_scope)) as Operation;

            // Tests whether initial_value_name exists first for backwards compatibility.
            if (!string.IsNullOrEmpty(variable_def.InitialValueName))
                _initial_value = g.as_graph_element(
                    ops.prepend_name_scope(variable_def.InitialValueName,
                                 import_scope: import_scope)) as Tensor;
            else
                _initial_value = null;

            _trainable = variable_def.Trainable;
            _snapshot = g.as_graph_element(
                ops.prepend_name_scope(variable_def.SnapshotName,
                               import_scope: import_scope)) as Tensor;

            if (variable_def.SaveSliceInfoDef != null)
                throw new NotImplementedException("save_slice_info_def");
            else
#pragma warning disable CS0642 // Possible mistaken empty statement
                ;// _save_slice_info = null;
#pragma warning restore CS0642 // Possible mistaken empty statement

            //_caching_device = null;
            //_constraint = null;
        }

        private void _init_from_args(object initial_value,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (initial_value is null)
                throw new ValueError("initial_value must be specified.");

            var init_from_fn = initial_value.GetType().Name == "Func`1";

            if (collections == null)
            {
                collections = new List<string> { tf.GraphKeys.GLOBAL_VARIABLES };
            }

            // Store the graph key so optimizers know how to only retrieve variables from
            // this graph.
            _graph_key = ops.get_default_graph().graph_key;

            _trainable = trainable;
            if (trainable && !collections.Contains(tf.GraphKeys.TRAINABLE_VARIABLES))
                collections.Add(tf.GraphKeys.TRAINABLE_VARIABLES);

            tf_with(ops.init_scope(), init_scope =>
            {
                var values = init_from_fn ? new object[0] : new object[] { initial_value };
                tf_with(ops.name_scope(name, "Variable", values), scope =>
                {
                    name = scope;

                    if (init_from_fn)
                    {
                        // Use attr_scope and device(None) to simulate the behavior of
                        // colocate_with when the variable we want to colocate with doesn't
                        // yet exist.
                        string true_name = ops.name_from_scope_name(name);
                        var attr = new AttrValue
                        {
                            List = new AttrValue.Types.ListValue()
                        };
                        attr.List.S.Add(ByteString.CopyFromUtf8($"loc:{true_name}"));
                        tf_with(ops.name_scope("Initializer"), scope2 =>
                        {
                            _initial_value = (initial_value as Func<Tensor>)();
                            _initial_value = ops.convert_to_tensor(_initial_value, name: "initial_value", dtype: dtype);
                        });
                        _variable = state_ops.variable_op_v2(_initial_value.shape, _initial_value.dtype.as_base_dtype(), name: name);
                    }
                    // Or get the initial value from a Tensor or Python object.
                    else
                    {
                        _initial_value = ops.convert_to_tensor(initial_value, name: "initial_value", dtype: dtype);

                        var shape = _initial_value.shape;
                        dtype = _initial_value.dtype;
                        _variable = gen_state_ops.variable_v2(shape, dtype.as_base_dtype(), scope);
                    }

                    // Manually overrides the variable's shape with the initial value's.
                    if (validate_shape)
                    {
                        var initial_value_shape = _initial_value.TensorShape;
                        if (!initial_value_shape.is_fully_defined())
                            throw new ValueError($"initial_value must have a shape specified: {_initial_value}");
                    }

                    // If 'initial_value' makes use of other variables, make sure we don't
                    // have an issue if these other variables aren't initialized first by
                    // using their initialized_value() method.
                    var _initial_value2 = _try_guard_against_uninitialized_dependencies(name, _initial_value);

                    _initializer_op = gen_state_ops.assign(_variable, _initial_value2, validate_shape).op;

                    if (!String.IsNullOrEmpty(caching_device))
                    {

                    }
                    else
                    {
                        ops.colocate_with(_initializer_op);

                        _snapshot = gen_array_ops.identity(_variable, name = "read");
                    }

                    ops.add_to_collections(collections, this as IVariableV1);
                });
            });
        }

        public Tensor _ref() => _variable;

        public Tensor value() => _snapshot;

        public Tensor AsTensor(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false) => _snapshot;

        public Tensor _as_graph_element() => _variable;

        public Tensor _TensorConversionFunction(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            if (as_ref)
                return _ref();
            else
                return value();
        }

        /// <summary>
        /// Attempt to guard against dependencies on uninitialized variables.
        /// </summary>
        /// <param name="initial_value"></param>
        private Tensor _try_guard_against_uninitialized_dependencies(string name, Tensor initial_value)
        {
            return _safe_initial_value_from_tensor(name, initial_value, op_cache: new Dictionary<string, Operation>());
        }

        /// <summary>
        /// Replace dependencies on variables with their initialized values.
        /// </summary>
        /// <param name="tensor">A `Tensor`. The tensor to replace.</param>
        /// <param name="op_cache">A dict mapping operation names to `Operation`s.</param>
        /// <returns>A `Tensor` compatible with `tensor`.</returns>
        private Tensor _safe_initial_value_from_tensor(string name, Tensor tensor, Dictionary<string, Operation> op_cache)
        {
            var op = tensor.op;
            var new_op = op_cache.ContainsKey(op.name) ? op_cache[op.name] : null;
            if (new_op == null)
            {
                new_op = _safe_initial_value_from_op(name, op, op_cache);
                op_cache[op.name] = new_op;
            }
            return new_op.outputs[tensor.value_index];
        }

        private Operation _safe_initial_value_from_op(string name, Operation op, Dictionary<string, Operation> op_cache)
        {
            var op_type = op.node_def.Op;
            switch (op_type)
            {
                case "IsVariableInitialized":
                case "VarIsInitializedOp":
                case "ReadVariableOp":
                    return op;
                case "Variable":
                case "VariableV2":
                case "VarHandleOp":
                    var initialized_value = _find_initialized_value_for_variable(op);
                    return initialized_value == null ? op : initialized_value.op;
            }

            // Recursively build initializer expressions for inputs.
            var modified = false;
            var new_op_inputs = new List<Tensor>();
            foreach (var op_input in op.inputs)
            {
                var new_op_input = _safe_initial_value_from_tensor(name, op_input as Tensor, op_cache);
                new_op_inputs.Add(new_op_input);
                modified = modified || new_op_input != op_input;
            }

            // If at least one input was modified, replace the op.
            if (modified)
            {
                var new_op_type = op_type;
                if (new_op_type == "RefSwitch")
                    new_op_type = "Switch";
                var new_op_name = op.node_def.Name + "_" + name;
                new_op_name = new_op_name.Replace(":", "_");

                // Convert attr values to AttrValue protos.
                var attr_protos = new Dictionary<string, AttrValue>();
                foreach (var attr_def in op.node_def.Attr)
                    attr_protos[attr_def.Key] = attr_def.Value;

                return op.graph.create_op(new_op_type, new_op_inputs.ToArray(), op._output_types,
                    name: new_op_name, attrs: attr_protos);
            }
            return op;
        }

        private Operation _find_initialized_value_for_variable(Operation variable_op)
        {
            var var_names = new[] { variable_op.node_def.Name, variable_op.node_def.Name + ":0" };
            foreach (var collection_name in new[]{tf.GraphKeys.GLOBAL_VARIABLES,
                            tf.GraphKeys.LOCAL_VARIABLES })
            {
                foreach (var var in variable_op.graph.get_collection<RefVariable>(collection_name))
                    if (var_names.Contains(var.Name))
                        return var.initialized_value();
            }

            return null;
        }

        /// <summary>
        /// Assigns a new value to the variable.
        /// </summary>
        /// <param name="value">The new value for this variable.</param>
        /// <param name="use_locking">If `True`, use locking during the assignment.</param>
        /// <param name="name">The name of the operation to be created</param>
        /// <param name="read_value">
        /// if True, will return something which evaluates to the
        /// new value of the variable; if False will return the assign op.
        /// </param>
        /// <returns>
        /// A `Tensor` that will hold the new value of this variable after
        /// the assignment has completed.
        /// </returns>
        public Tensor assign<T>(T value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var assign = gen_state_ops.assign(_variable, value, use_locking: use_locking, name: name);
            if (read_value)
                return assign;
            return assign.op;
        }
        
        public override string ToString()
        {
            return $"tf.RefVariable '{Name}' shape={shape} dtype={dtype}";
        }

        public VariableDef to_proto(string export_scope)
        {
            if (string.IsNullOrEmpty(export_scope) || _variable.name.StartsWith(export_scope))
            {
                var var_def = new VariableDef();
                var_def.VariableName = ops.strip_name_scope(_variable.name, export_scope);
                if (_initial_value != null)
                    var_def.InitialValueName = ops.strip_name_scope(_initial_value.name, export_scope);
                var_def.Trainable = _trainable;
                var_def.InitializerName = ops.strip_name_scope(Initializer.name, export_scope);
                var_def.SnapshotName = ops.strip_name_scope(_snapshot.name, export_scope);
                if (_save_slice_info)
                    throw new NotImplementedException("to_proto _save_slice_info");

                return var_def;
            }

            throw new NotImplementedException("to_proto RefVariable");
        }

        public RefVariable from_proto(VariableDef proto, string import_scope)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the value of this variable, read in the current context.
        /// </summary>
        /// <returns></returns>
        private ITensorOrOperation read_value()
        {
            return array_ops.identity(_variable, name: "read");
        }

        /// <summary>
        /// Returns the Tensor used as the initial value for the variable.
        /// </summary>
        /// <returns></returns>
        private ITensorOrOperation initial_value()
        {
            return _initial_value;
        }

        public Tensor is_variable_initialized(RefVariable variable)
        {
            return state_ops.is_variable_initialized(variable);
        }

        public Tensor initialized_value()
        {
            ops.init_scope();
            return control_flow_ops.cond(is_variable_initialized(this),
                                   read_value,
                                   initial_value);
        }

        //  Update 'ref' by adding 'value' to it.
        //  This operation outputs "ref" after the update is done.
        //  This makes it easier to chain operations that need to use the reset value.
        //  Args:
        //    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
        //      Should be from a `Variable` node.
        //    value: A `Tensor`. Must have the same type as `ref`.
        //      The value to be added to the variable.
        //    use_locking: An optional `bool`. Defaults to `False`.
        //      If True, the addition will be protected by a lock;
        //        otherwise the behavior is undefined, but may exhibit less contention.
        //      name: A name for the operation(optional).
        //  Returns:
        //    A mutable `Tensor`. Has the same type as `ref`.
        public Tensor assign_add<T>(T value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var variable = this;
            var _op = tf.OpDefLib._apply_op_helper("AssignAdd", name: name, args: new { variable, value, use_locking });
            return _op;
        }

        public NDArray numpy()
            => throw new RuntimeError("Graph mode can't use numpy().");

        public Tensor assign_sub<T>(T delta, bool use_locking = false, string name = null, bool read_value = true)
        {
            throw new NotImplementedException();
        }

        public IVariableV1 assign_sub_lazy_load(Tensor delta, string name = null)
        {
            throw new NotImplementedException();
        }

        public IVariableV1 assign_lazy_load(Tensor value, string name = null)
        {
            throw new NotImplementedException();
        }
    }
}
