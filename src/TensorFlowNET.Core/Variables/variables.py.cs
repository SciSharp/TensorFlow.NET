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
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class variables
    {
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        public static object trainable_variables()
        {
            return ops.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES);
        }

        /// <summary>
        /// Returns all variables and `SaveableObject`s that must be checkpointed.
        /// </summary>
        /// <param name="scope"></param>
        /// <returns></returns>
        public static VariableV1[] _all_saveable_objects(string scope = "")
        {
            var all = new List<VariableV1>();

            var collection = ops.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope);
            if(collection != null)
                all.AddRange(collection as List<VariableV1>);

            collection = ops.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, scope);
            if (collection != null)
                all.AddRange(collection as List<VariableV1>);

            return all.ToArray();
        }

        /// <summary>
        /// Returns global variables.
        /// </summary>
        /// <param name="scope">
        /// (Optional.) A string. If supplied, the resulting list is filtered
        /// to include only items whose `name` attribute matches `scope` using
        /// `re.match`. Items without a `name` attribute are never returned if a
        /// scope is supplied. The choice of `re.match` means that a `scope` without
        /// special tokens filters by prefix.
        /// </param>
        /// <returns>A list of `Variable` objects.</returns>
        public static List<VariableV1> global_variables(string scope = null)
        {
            var result = ops.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope);

            return result == null ? new List<VariableV1>() : result as List<VariableV1>;
        }

        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        public static Operation variables_initializer(VariableV1[] var_list, string name = "init")
        {
            if (var_list.Length > 0)
                return control_flow_ops.group(var_list.Select(x => x.initializer).ToArray(), name);
            else
                return gen_control_flow_ops.no_op(name: name);
        }

        public static Tensor _try_guard_against_uninitialized_dependencies(string name, Tensor initial_value)
        {
            return _safe_initial_value_from_tensor(name, initial_value, new Dictionary<string, Operation>());
        }

        public static Tensor _safe_initial_value_from_tensor(string name, Tensor tensor, Dictionary<string, Operation> op_cache)
        {
            var op = tensor.op;
            Operation new_op = op_cache.ContainsKey(op.name) ? op_cache[op.name] : null;
            if(new_op == null)
            {
                new_op = _safe_initial_value_from_op(name, op, op_cache);
                op_cache[op.name] = new_op;
            }

            return new_op.outputs[tensor.value_index];
        }

        /// <summary>
        /// Replace dependencies on variables with their initialized values.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="op"></param>
        /// <param name="op_cache"></param>
        /// <returns></returns>
        public static Operation _safe_initial_value_from_op(string name, Operation op, Dictionary<string, Operation> op_cache)
        {
            var op_type = op.node_def.Op;
            if (op_type == "IsVariableInitialized" ||
                op_type == "VarIsInitializedOp" ||
                op_type == "ReadVariableOp")
                return op;

            if(op_type == "Variable" ||
                op_type == "VariableV2" ||
                op_type == "VarHandleOp")
            {
                throw new NotImplementedException("");
            }

            // Recursively build initializer expressions for inputs.
            bool modified = false;
            var new_op_inputs = new List<Tensor>();
            foreach(Tensor op_input in op.inputs)
            {
                var new_op_input = _safe_initial_value_from_tensor(name, op_input, op_cache);
                new_op_inputs.Add(new_op_input);
                modified = modified || new_op_input != op_input;
            }

            // If at least one input was modified, replace the op.
            return op;
        }

        public static Tensor global_variables_initializer()
        {
            throw new NotImplementedException();
        }
    }
}
