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

using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Utils
{
    public class base_layer_utils
    {
        /// <summary>
        /// Adds a new variable to the layer.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        public static IVariableV1 make_variable(VariableArgs args)
        {
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            var initializing_from_value = false;
#pragma warning restore CS0219 // Variable is assigned but its value is never used

            Func<Tensor> init_val = () => args.Initializer.Apply(new InitializerArgs(args.Shape, dtype: args.DType));

            var variable_dtype = args.DType.as_base_dtype();
            return tf.Variable(init_val,
                dtype: variable_dtype,
                shape: args.Shape,
                name: args.Name,
                trainable: args.Trainable,
                validate_shape: args.ValidateShape,
                use_resource: args.UseResource);
        }

        /// <summary>
        /// Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static string unique_layer_name(string name, Dictionary<string, int> name_uid_map = null,
            string[] avoid_names = null, bool zero_based = false)
        {
            if (name_uid_map == null)
                name_uid_map = get_default_graph_uid_map();
            if (avoid_names == null)
                avoid_names = new string[0];

            string proposed_name = null;
            while (proposed_name == null || avoid_names.Contains(proposed_name))
            {
                if (!name_uid_map.ContainsKey(name))
                    name_uid_map[name] = 0;

                if (zero_based)
                {
                    int number = name_uid_map[name];
                    if (number > 0)
                        proposed_name = $"{name}_{number}";
                    else
                        proposed_name = name;

                    name_uid_map[name] += 1;
                }
                else
                {
                    name_uid_map[name] += 1;
                    proposed_name = $"{name}_{name_uid_map[name]}";
                }
            }

            return proposed_name;
        }

        public static Dictionary<string, int> get_default_graph_uid_map()
        {
            var graph = ops.get_default_graph();
            Dictionary<string, int> name_uid_map = null;
            if (keras.backend.PER_GRAPH_LAYER_NAME_UIDS.ContainsKey(graph))
            {
                name_uid_map = keras.backend.PER_GRAPH_LAYER_NAME_UIDS[graph];
            }
            else
            {
                name_uid_map = new Dictionary<string, int>();
                keras.backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map;
            }

            return name_uid_map;
        }

        public static bool needs_keras_history(Tensors inputs)
        {
            if (inputs.Any(x => x.KerasHistory == null))
                return true;

            return false;
        }

        public static Layer[] create_keras_history(Tensors inputs)
        {
            var processed_ops = new List<Operation>();
            var created_layers = new List<Layer>();
            CreateKerasHistoryHelper(inputs, processed_ops, created_layers);
            return created_layers.ToArray();
        }

        public static void CreateKerasHistoryHelper(Tensors tensors, List<Operation> processed_ops, List<Layer> created_layers)
        {
            foreach (var tensor in tensors)
            {
                if (tensor.KerasHistory != null)
                    continue;

                var op = tensor.op;
                if (!processed_ops.Contains(op))
                {
                    var layer_inputs = new List<Tensor>();
                    var constants = new Dictionary<int, NDArray>();
                    foreach (var (i, op_input) in enumerate(op.inputs._inputs))
                    {
                        if (uses_keras_history(op_input))
                            layer_inputs.Add(op_input);
                        else
                        {
                            tf_with(ops.init_scope(), delegate
                            {
                                constants[i] = keras.backend.eval_in_eager_or_function(op_input);
                            });
                        }
                    }

                    // recursively
                    CreateKerasHistoryHelper(layer_inputs, processed_ops, created_layers);
                    var opLayerArgs = new TensorFlowOpLayerArgs
                    {
                        NodeDef = op.node_def,
                        Constants = constants,
                        Name = op.name
                    };
                    var op_layer = new TensorFlowOpLayer(opLayerArgs);
                    created_layers.Add(op_layer);
                    op_layer.SetConnectivityMetadata(layer_inputs, op.outputs);
                    processed_ops.Add(op);
                }
            }
        }

        // recusive
        static bool uses_keras_history(Tensor op_input)
        {
            if (op_input.KerasHistory != null)
                return true;

            foreach (var input in op_input.op.inputs._inputs)
                if (uses_keras_history(input))
                    return true;

            return false;
        }
    }
}
