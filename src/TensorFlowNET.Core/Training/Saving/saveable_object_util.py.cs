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
using System.Diagnostics;
using System.Linq;
using Tensorflow.Checkpoint;
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

        private static void _add_saveable<T>(List<T> saveables, List<Tensor> seen_ops, T saveable) where T : MySaveableObject
        {
            if (seen_ops.Contains(saveable.op))
                throw new ValueError($"The same saveable will be restored with two names: {saveable.name}");

            saveables.Add(saveable);
            seen_ops.Add(saveable.op);
        }

        /// <summary>
        /// Create `SaveableObject`s from an operation. Note that the `op` should not be implicitly converted from `Variable`.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static IEnumerable<MySaveableObject> saveable_objects_for_op(Tensor op, string name)
        {
            if (false)
            {

            }
            else
            {
                ops.init_scope();
                var variable = ops.convert_to_tensor(op, as_ref: true);
                if (variable.dtype.is_ref_dtype())
                    yield return new ReferenceVariableSaveable(variable, "", name);
                else
                    yield return new ResourceVariableSaveable(variable, "", name);
            }
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
                    Debug.Assert(variable is ResourceVariable);
                    yield return new ResourceVariableSaveable((ResourceVariable)variable, "", name);
                }
            }
            else
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
                    if(factory.DataType == typeof(ResourceVariable))
                    {
                        var variable = factory.GetValueA();
                        foreach (var op in saveable_objects_for_op(variable as Trackable, variable.Name))
                        {
                            yield return op;
                        }
                    }
                    else
                    {
                        var variable = factory.GetValueB();
                        foreach (var op in saveable_objects_for_op(variable, variable.name))
                        {
                            yield return op;
                        }
                    }
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

        public static IDictionary<string, Maybe<ResourceVariable, MySaveableObject>> saveable_objects_from_trackable(Trackable obj)
        {
            // skip the process of type `PythonState`

            if (trackable_has_serialize_to_tensor(obj))
            {
                var name = TrackableUtils.SERIALIZE_TO_TENSORS_NAME;
                // skip the case that `obj._serialize_to_tensors` is `ConcreteFunction`.
                var tensor_dict = obj.serialize_to_tensors();

                List<SaveSpec> specs = new();
                List<string> local_names = new();
                string prefix = SaveableCompat.get_saveable_name(obj) ?? "";
                foreach(var pair in tensor_dict)
                {
                    var tensor_name = pair.Key;
                    var maybe_tensor = pair.Value;
                    local_names.Add(tensor_name);
                    string spec_name = name + TrackableUtils.escape_local_name(tensor_name);

                    IDictionary<string, Tensor> internal_dict;
                    if(maybe_tensor.DataType == typeof(Tensor))
                    {
                        internal_dict= new Dictionary<string, Tensor>();
                        internal_dict[""] = maybe_tensor.GetValueA();
                    }
                    else
                    {
                        internal_dict = maybe_tensor.GetValueB();
                    }

                    foreach(var item in internal_dict)
                    {
                        specs.Add(new SaveSpec(item.Value, item.Key, spec_name));
                    }
                }
                Dictionary<string, Maybe<ResourceVariable, MySaveableObject>> res = new();
                res[name] = new TrackableSaveable(obj, specs, name, local_names, prefix);
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
    }

    public class SaveableCompatibilityConverter: Trackable
    {
        private Trackable _obj;
        private IList<MySaveableObject> _saveables;
        public SaveableCompatibilityConverter(Trackable obj, IList<MySaveableObject> saveables)
        {
            _obj= obj;
            _saveables= saveables;
        }

        public Trackable Obj => _obj;
        public IList<MySaveableObject> mySaveables=> _saveables;

        public override IDictionary<string, Maybe<Tensor, IDictionary<string, Tensor>>> serialize_to_tensors()
        {
            return saveable_object_to_tensor_dict(_saveables);
        }

        /// <summary>
        /// Converts a list of SaveableObjects to a tensor dictionary.
        /// </summary>
        /// <param name="saveables"></param>
        public static Dictionary<string, Maybe<Tensor, IDictionary<string, Tensor>>> saveable_object_to_tensor_dict(IList<MySaveableObject> saveables)
        {
            Dictionary<string, Maybe<Tensor, IDictionary<string, Tensor>>> tensor_dict = new();
            foreach (var saveable in saveables)
            {
                foreach(var spec in saveable.specs)
                {
                    // skip the check that if `spec` is callable.
                    var name = saveable_object_util.convert_to_string(spec.name);
                    var slice_spec = saveable_object_util.convert_to_string(spec.slice_spec);
                    if (!string.IsNullOrEmpty(slice_spec))
                    {
                        tensor_dict.SetDefault(name, new Dictionary<string, Tensor>()).GetValueB()[slice_spec] = spec.tensor;
                    }
                    else
                    {
                        tensor_dict[name] = spec.tensor;
                    }
                }
            }
            return tensor_dict;
        }
    }
}
