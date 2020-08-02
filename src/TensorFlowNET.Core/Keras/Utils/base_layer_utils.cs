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

            ops.init_scope();

            Func<Tensor> init_val = () => args.Initializer.Apply(new InitializerArgs
            {
                Shape = args.Shape,
                DType = args.DType
            });

            var variable_dtype = args.DType.as_base_dtype();
            var v = tf.Variable(init_val,
                dtype: variable_dtype,
                shape: args.Shape,
                name: args.Name,
                trainable: args.Trainable,
                validate_shape: args.ValidateShape,
                use_resource: args.UseResource);

            return v;
        }

        /// <summary>
        /// Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static string unique_layer_name(string name, Dictionary<(string, string), int> name_uid_map = null,
            string[] avoid_names = null, string @namespace = "", bool zero_based = false)
        {
            if (name_uid_map == null)
                name_uid_map = get_default_graph_uid_map();
            if (avoid_names == null)
                avoid_names = new string[0];

            string proposed_name = null;
            while (proposed_name == null || avoid_names.Contains(proposed_name))
            {
                var name_key = (@namespace, name);
                if (!name_uid_map.ContainsKey(name_key))
                    name_uid_map[name_key] = 0;

                if (zero_based)
                {
                    int number = name_uid_map[name_key];
                    if (number > 0)
                        proposed_name = $"{name}_{number}";
                    else
                        proposed_name = name;

                    name_uid_map[name_key] += 1;
                }
                else
                {
                    name_uid_map[name_key] += 1;
                    proposed_name = $"{name}_{name_uid_map[name_key]}";
                }
            }

            return proposed_name;
        }

        public static Dictionary<(string, string), int> get_default_graph_uid_map()
        {
            var graph = ops.get_default_graph();
            Dictionary<(string, string), int> name_uid_map = null;
            if (tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS.ContainsKey(graph))
            {
                name_uid_map = tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS[graph];
            }
            else
            {
                name_uid_map = new Dictionary<(string, string), int>();
                tf.keras.backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map;
            }

            return name_uid_map;
        }
    }
}
