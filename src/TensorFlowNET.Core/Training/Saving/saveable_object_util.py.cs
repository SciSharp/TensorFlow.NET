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

using OneOf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Checkpoint;
using Tensorflow.Contexts;
using Tensorflow.Device;
using Tensorflow.Operations.Activation;
using Tensorflow.Train;
using Tensorflow.Training;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A SaveableObject that defines `Trackable` checkpointing steps.
    /// </summary>
    public class TrackableSaveable : MySaveableObject
    {
        private string _prefix;
        private IEnumerable<string> _local_names;
        private Trackable _trackable;
        private bool _call_with_mapped_captures;
        // TODO: revise the implementation. Currently the parameter of constructor of this class and its base class has conflict.
        public TrackableSaveable(Trackable obj, IEnumerable<SaveSpec> specs, string name, IEnumerable<string> local_names,
            string prefix, bool call_with_mapped_captures = false) : base((object)obj as Tensor, specs.ToArray(), name)
        {
            _prefix = prefix;
            _trackable = obj;
            _local_names = local_names;
            _call_with_mapped_captures = call_with_mapped_captures;
        }

        // TODO: complete this class.
    }
    public static class saveable_object_util
    {
        public static string NO_SLICE_SPEC_KEY = "";
        private static HashSet<string> _VARIABLE_OPS = new HashSet<string>(new string[] {
            "Variable", "VariableV2", "AutoReloadVariable", "VarHandleOp", "ReadVariableOp"
        });
        /// <summary>
        /// Returns the variables and names that will be used for a Saver.
        /// </summary>
        /// <param name="names_to_saveables"></param>
        /// <returns></returns>
        public static MySaveableObject[] validate_and_slice_inputs(IVariableV1[] names_to_saveables)
        {
            var names_to_saveables_dict = op_list_to_dict(names_to_saveables);
            var saveables = new List<MySaveableObject>();
            var seen_ops = new List<Tensor>();

            foreach (var (name, op) in enumerate(names_to_saveables_dict))
            {
                foreach (var converted_saveable_object in saveable_objects_for_op(op, name))
                    _add_saveable(saveables, seen_ops, converted_saveable_object);
            }
            return saveables.ToArray();
        }

        public static MySaveableObject[] validate_and_slice_inputs(Dictionary<string, Tensor> names_to_saveables)
        {
            var saveables = new List<MySaveableObject>();
            var seen_ops = new List<Tensor>();

            foreach (var (name, op) in enumerate(names_to_saveables))
            {
                foreach (var converted_saveable_object in saveable_objects_for_op(op, name))
                    _add_saveable(saveables, seen_ops, converted_saveable_object);
            }
            return saveables.ToArray();
        }

        public static MySaveableObject[] validate_and_slice_inputs(Dictionary<string, BaseResourceVariable> names_to_saveables)
        {
            var saveables = new List<MySaveableObject>();
            var seen_ops = new List<BaseResourceVariable>();

            foreach(var item in names_to_saveables.OrderBy(x => x.Key))
            {
                foreach(var converted_saveable_object in saveable_objects_for_op(item.Value, item.Key))
                {
                    _add_saveable(saveables, seen_ops, converted_saveable_object);
                }
            }
            return saveables.ToArray();
        }

        private static void _add_saveable<T>(List<T> saveables, List<Tensor> seen_ops, T saveable) where T : MySaveableObject
        {
            if (seen_ops.Contains(saveable.op))
                throw new ValueError($"The same saveable will be restored with two names: {saveable.name}");

            saveables.Add(saveable);
            seen_ops.Add(saveable.op);
        }

        private static void _add_saveable(List<MySaveableObject> saveables, List<BaseResourceVariable> seen_ops, MySaveableObject saveable)
        {
            if (seen_ops.Contains(saveable.variable))
                throw new ValueError($"The same saveable will be restored with two names: {saveable.op.OriginalVar.Name}");

            saveables.Add(saveable);
            seen_ops.Add(saveable.variable);
        }

        /// <summary>
        /// Create `SaveableObject`s from an operation. Note that the `op` should not be implicitly converted from `Variable`.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static IEnumerable<MySaveableObject> saveable_objects_for_op(Tensor op, string name)
        {
            ops.init_scope();
            var variable = ops.convert_to_tensor(op, as_ref: true);
            if (variable.dtype.is_ref_dtype())
                yield return new ReferenceVariableSaveable(variable, "", name);
            else
                yield return new ResourceVariableSaveable(variable, "", name);
        }

        /// <summary>
        /// Create `SaveableObject`s from an operation.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static IEnumerable<MySaveableObject> saveable_objects_for_op(Trackable obj, string name)
        {
            // The `op` maybe `Variable` or `Trackable`.
            if (obj is BaseResourceVariable)
            {
                var variable = obj as BaseResourceVariable;
                if (variable.InGraphMode)
                {
                    yield return new ResourceVariableSaveable(variable.GraphElement, "", name);
                }
                else
                {
                    yield return new ResourceVariableSaveable(variable, "", name);
                }
            }
            else if(obj is not IVariableV1)
            {
                foreach(var pair in saveable_objects_from_trackable(obj))
                {
                    var attr = pair.Key;
                    var factory = pair.Value;
                    string full_name;
                    if(attr == Trackable.Constants.VARIABLE_VALUE_KEY)
                    {
                        full_name = name;
                    }
                    else
                    {
                        full_name = name + "_" + attr;
                    }
                    var op = factory(full_name);
                    if(op.TryPickT0(out var variable, out var saveable))
                    {
                        foreach (var v in saveable_objects_for_op(variable as Trackable, variable.Name))
                        {
                            yield return v;
                        }
                    }
                    else
                    {
                        foreach (var v in saveable_objects_for_op(saveable, saveable.name))
                        {
                            yield return v;
                        }
                    }
                }
            }
            else
            {
                // Variable
                if (tf.Context.executing_eagerly())
                {
                    throw new ValueError($"Can only save/restore ResourceVariables when " +
                        $"executing eagerly, got type: {obj.GetType()}.");
                }
                var variable = ops.convert_to_tensor(obj, as_ref: true);
                if (!_tensor_comes_from_variable(variable))
                {
                    throw new TypeError($"names_to_saveables must be a dict mapping string " +
                        $"names to Tensors/Variables. Not a variable: {variable}");
                }
                if(variable.op.type == "Variable" || variable.op.type == "VariableV2" ||
                    variable.op.type == "AutoReloadVariable")
                {
                    yield return new ReferenceVariableSaveable(variable, "", name);
                }
                else
                {
                    yield return new ResourceVariableSaveable(variable, "", name);
                }
            }
        }

        /// <summary>
        /// Create `SaveableObject`s from an operation.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static IEnumerable<MySaveableObject> saveable_objects_for_op(MySaveableObject obj, string name)
        {
            yield return obj;
        }

        public static Dictionary<string, Tensor> op_list_to_dict(IVariableV1[] op_list, bool convert_variable_to_tensor = true)
        {
            op_list = op_list.OrderBy(x => x.Name).ToArray();
            var names_to_saveables = new Dictionary<string, Tensor>();

            foreach (var var in op_list)
            {
                bool resource_or_ref_variable = var is RefVariable || var is ResourceVariable;
                if (false)
                {
                    throw new NotImplementedException("op_list_to_dict");
                }
                else
                {
                    // Variables (reference and resource) have an _in_graph_mode property
                    if (false) // eager
                    {

                    }
                    else
                    {
                        string name = null;
                        Tensor tensor = null;

                        if (convert_variable_to_tensor)
                        {
                            if (!var.dtype.is_ref_dtype())
                                tensor = var.GraphElement;
                            else
                                tensor = ops.convert_to_tensor(var, as_ref: true);
                        }

                        if (tensor.op.type == "ReadVariableOp")
                            name = tensor.op.inputs[0].op.name;
                        else
                            name = var.Op.name;

                        if (names_to_saveables.ContainsKey(name))
                            throw new ValueError($"At least two variables have the same name: {name}");

                        names_to_saveables[name] = tensor;
                    }
                }
            }

            return names_to_saveables;
        }

        public static IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> saveable_objects_from_trackable(Trackable obj)
        {
            // skip the process of type `PythonState`

            OneOf<BaseResourceVariable, MySaveableObject> create_saveable(string name = "")
            {
                // skip the case that `obj._serialize_to_tensors` is `ConcreteFunction`.
                var tensor_dict = obj.serialize_to_tensors();

                List<SaveSpec> specs = new();
                List<string> local_names = new();
                string prefix = SaveableCompat.get_saveable_name(obj) ?? "";
                foreach (var pair in tensor_dict)
                {
                    var tensor_name = pair.Key;
                    var internal_dict = pair.Value;
                    local_names.Add(tensor_name);
                    string spec_name = name + TrackableUtils.escape_local_name(tensor_name);

                    foreach (var item in internal_dict)
                    {
                        Debug.Assert(item.Value.IsT0);
                        specs.Add(new SaveSpec(item.Value.AsT0, item.Key, spec_name));
                    }
                }
                return new TrackableSaveable(obj, specs, name, local_names, prefix);
            }

            if (trackable_has_serialize_to_tensor(obj))
            {
                Dictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> res = new();
                res[TrackableUtils.SERIALIZE_TO_TENSORS_NAME] = create_saveable;
                return res;
            }
            else
            {
                return obj.gather_saveables_for_checkpoint();
            }
        }

        public static bool trackable_has_serialize_to_tensor(Trackable obj)
        {
            return obj.GetType().GetMethod("serialize_to_tensors").DeclaringType != typeof(Trackable);
        }

        internal static string convert_to_string(string x)
        {
            return tf.compat.as_str(x);
        }

        /// <summary>
        /// Converts a list of SaveableObjects to a tensor dictionary.
        /// </summary>
        /// <param name="saveables"></param>
        public static Dictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> saveable_object_to_tensor_dict(IList<MySaveableObject> saveables)
        {
            Dictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> tensor_dict = new();
            foreach (var saveable in saveables)
            {
                foreach (var spec in saveable.specs)
                {
                    // skip the check that if `spec` is callable.
                    var name = convert_to_string(spec.name);
                    var slice_spec = convert_to_string(spec.slice_spec);
                    if (string.IsNullOrEmpty(slice_spec))
                    {
                        slice_spec = NO_SLICE_SPEC_KEY;
                    }
                    tensor_dict.SetDefault(name, new Dictionary<string, OneOf<Tensor, SaveSpec>>())[slice_spec] = spec.TensorCreator is null ? spec.tensor : spec;
                }
            }
            return tensor_dict;
        }

        /// <summary>
        /// Generates `Trackable._restore_from_tensors` from SaveableObjects.
        /// </summary>
        /// <returns></returns>
        public static Func<IDictionary<string, OneOf<Tensor, IDictionary<string, Tensor>>>, IDictionary<string, Operation>> saveable_object_to_restore_fn(IList<MySaveableObject> saveables)
        {
            return (restored_tensors) =>
            {
                Dictionary<string, Operation> restored_ops = new();

                foreach(var saveable in saveables)
                {
                    List<Tensor> saveable_restored_tensors = new();
                    foreach(var spec in saveable.specs)
                    {
                        var name = TrackableUtils.extract_local_name(saveable_object_util.convert_to_string(spec.name));
                        var slice_spec = saveable_object_util.convert_to_string(spec.slice_spec);

                        var maybe_tensor = restored_tensors[name];
                        IDictionary<string, Tensor> dict;
                        if(maybe_tensor.TryPickT0(out var tensor, out var dic))
                        {
                            dict = new Dictionary<string, Tensor>();
                            dict[""] = tensor;
                        }
                        else
                        {
                            dict = dic;
                        }
                        saveable_restored_tensors.Add(dict[slice_spec]);
                    }
                    restored_ops[saveable.name] = saveable.restore(saveable_restored_tensors.ToArray(), null);
                }
                return restored_ops;
            };
        }

        /// <summary>
        /// Returns a dict of SaveableObject factories generated from loaded fns.
        /// </summary>
        /// <param name="saveable_fn_by_name"></param>
        /// <param name="temp_session"></param>
        public static IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> recreate_saveable_objects(
            IDictionary<string, (Trackable, Trackable)> saveable_fn_by_name, IEnumerable<object>? temp_session)
        {
            if (saveable_fn_by_name.Count > 0)
            {
                throw new NotImplementedException("Not implemented, please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
            }
            var res = new Dictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>>();
            return res;
        }

        public static OneOf<BaseResourceVariable, MySaveableObject> create_saveable_object(string name, string key, Func<string, OneOf<BaseResourceVariable, MySaveableObject>> factory, 
            bool call_with_mapped_captures = false)
        {
            return factory(key);
        }

        public static string set_cpu0(string device_string)
        {
            if (tf.Context.is_custom_device(device_string))
            {
                return device_string;
            }
            var parsed_device = DeviceSpec.from_string(device_string);
            parsed_device = parsed_device.replace(device_type: "CPU", device_index: 0);
            return parsed_device.ToString();
        }

        private static bool _tensor_comes_from_variable(object v)
        {
            return v is Tensor tensor && _VARIABLE_OPS.Contains(tensor.op.type);
        }
    }

    public class SaveableCompatibilityConverter: Trackable
    {
        private object _obj;
        private IList<MySaveableObject> _saveables;
        public SaveableCompatibilityConverter(object obj, IList<MySaveableObject> saveables)
        {
            _obj= obj;
            _saveables= saveables;
        }

        public object Obj => _obj;
        public IList<MySaveableObject> mySaveables=> _saveables;

        public override IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> serialize_to_tensors()
        {
            return saveable_object_util.saveable_object_to_tensor_dict(_saveables);
        }

        /// <summary>
        /// Returns the restore ops defined in the Saveables.
        /// </summary>
        /// <param name="restored_tensors"></param>
        /// <returns></returns>
        public override IDictionary<string, Operation> _restore_from_tensors(IDictionary<string, OneOf<Tensor, IDictionary<string, Tensor>>> restored_tensors)
        {
            List<string> expected_keys = new();
            foreach(var saveable in _saveables)
            {
                expected_keys.AddRange(saveable.specs.Select(x => TrackableUtils.extract_local_name(saveable_object_util.convert_to_string(x.name))));
            }
            if (!expected_keys.Distinct().SequenceEqual(restored_tensors.Keys))
            {
                throw new ValueError($"Could not restore object {_obj} because not all expected tensors were in the checkpoint." +
                    $"\n\tExpected: {expected_keys} \n\tGot: {list(restored_tensors.Keys)}");
            }
            return saveable_object_util.saveable_object_to_restore_fn(_saveables).Invoke(restored_tensors);
        }
    }
}
