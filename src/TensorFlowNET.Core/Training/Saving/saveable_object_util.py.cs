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

using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public static class saveable_object_util
    {
        public class TrackableSaveable: MySaveableObject
        {
            
        }
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
        /// Create `SaveableObject`s from an operation.
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

        public static IDictionary<string, ResourceVariable> saveable_objects_from_trackable(Trackable obj)
        {
            // TODO: complete the implementation.
            return obj.gather_saveables_for_checkpoint();
        }

        public static bool trackable_has_serialize_to_tensor(Trackable obj)
        {
            // TODO: implement it.
            return false;
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

        public override IDictionary<string, object> serialize_to_tensors()
        {
            return saveable_objects_to_tensor_dict(_saveables);
        }

        /// <summary>
        /// Converts a list of SaveableObjects to a tensor dictionary.
        /// </summary>
        /// <param name="saveables"></param>
        public static Dictionary<string, object> saveable_objects_to_tensor_dict(IList<MySaveableObject> saveables)
        {
            Dictionary<string, object> tensor_dict = new();
            foreach (var saveable in saveables)
            {
                foreach(var spec in saveable.specs)
                {
                    var name = saveable_object_util.convert_to_string(spec.name);
                    var slice_spec = saveable_object_util.convert_to_string(spec.slice_spec);
                    if (!string.IsNullOrEmpty(slice_spec))
                    {
                        throw new NotImplementedException();
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
