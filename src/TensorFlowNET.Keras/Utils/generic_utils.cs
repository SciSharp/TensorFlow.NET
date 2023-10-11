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

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Security.AccessControl;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Saving;
using Tensorflow.Train;
using System.Text.RegularExpressions;

namespace Tensorflow.Keras.Utils
{
    public class generic_utils
    {
        private static readonly string _LAYER_UNDEFINED_CONFIG_KEY = "layer was saved without config";
        /// <summary>
        /// This method does not have corresponding method in python. It's close to `serialize_keras_object`.
        /// </summary>
        /// <param name="instance"></param>
        /// <returns></returns>
        public static LayerConfig serialize_layer_to_config(ILayer instance)
        {
            var config = instance.get_config();
            Debug.Assert(config is LayerArgs);
            return new LayerConfig
            {
                Config = config as LayerArgs,
                ClassName = instance.GetType().Name
            };
        }

        public static JObject serialize_keras_object(IKerasConfigable instance)
        {
            var config = JToken.FromObject(instance.get_config());
            // TODO: change the class_name to registered name, instead of system class name.
            return serialize_utils.serialize_keras_class_and_config(instance.GetType().Name, config, instance);
        }

        public static Layer deserialize_keras_object(string class_name, JToken config)
        {
            var argType = Assembly.Load("Tensorflow.Binding").GetType($"Tensorflow.Keras.ArgsDefinition.{class_name}Args");
            if(argType is null)
            {
                return null;
            }
            var deserializationMethod = typeof(JToken).GetMethods(BindingFlags.Instance | BindingFlags.Public)
                .Single(x => x.Name == "ToObject" && x.IsGenericMethodDefinition && x.GetParameters().Count() == 0);
            var deserializationGenericMethod = deserializationMethod.MakeGenericMethod(argType);
            var args = deserializationGenericMethod.Invoke(config, null);
            var layer = Assembly.Load("Tensorflow.Keras").CreateInstance($"Tensorflow.Keras.Layers.{class_name}", true, BindingFlags.Default, null, new object[] { args }, null, null);
            Debug.Assert(layer is Layer);

            // TODO(Rinne): _shared_object_loading_scope().set(shared_object_id, deserialized_obj)

            return layer as Layer;
        }

        public static Layer deserialize_keras_object(string class_name, LayerArgs args)
        {
            var layer = Assembly.Load("Tensorflow.Keras").CreateInstance($"Tensorflow.Keras.Layers.{class_name}", true, BindingFlags.Default, null, new object[] { args }, null, null);
            if (layer is null)
            {
                return null;
            }
            Debug.Assert(layer is Layer);

            // TODO(Rinne): _shared_object_loading_scope().set(shared_object_id, deserialized_obj)

            return layer as Layer;
        }

        public static LayerArgs deserialize_layer_args(string class_name, JToken config)
        {
            var argType = Assembly.Load("Tensorflow.Binding").GetType($"Tensorflow.Keras.ArgsDefinition.{class_name}Args");
            var deserializationMethod = typeof(JToken).GetMethods(BindingFlags.Instance | BindingFlags.Public)
                .Single(x => x.Name == "ToObject" && x.IsGenericMethodDefinition && x.GetParameters().Count() == 0);
            var deserializationGenericMethod = deserializationMethod.MakeGenericMethod(argType);
            var args = deserializationGenericMethod.Invoke(config, null);
            Debug.Assert(args is LayerArgs);
            return args as LayerArgs;
        }

        public static FunctionalConfig deserialize_model_config(JToken json)
        {
            FunctionalConfig config = new FunctionalConfig();
            config.Name = json["name"].ToObject<string>();
            config.Layers = new List<LayerConfig>();
            var layersToken = json["layers"];
            foreach (var token in layersToken)
            {
                var args = deserialize_layer_args(token["class_name"].ToObject<string>(), token["config"]);

                List<NodeConfig> nodeConfig = null; //python tensorflow sometimes exports inbound nodes in an extra nested array
                if (token["inbound_nodes"].Count() > 0 && token["inbound_nodes"][0].Count() > 0 && token["inbound_nodes"][0][0].Count() > 0)
                {
                    nodeConfig = token["inbound_nodes"].ToObject<List<List<NodeConfig>>>().FirstOrDefault() ?? new List<NodeConfig>();
                }
                else
                {
                    nodeConfig = token["inbound_nodes"].ToObject<List<NodeConfig>>();
                }

                config.Layers.Add(new LayerConfig()
                {
                    Config = args, 
                    Name = token["name"].ToObject<string>(), 
                    ClassName = token["class_name"].ToObject<string>(), 
                    InboundNodes = nodeConfig,
                });
            }
            config.InputLayers = json["input_layers"].ToObject<List<NodeConfig>>();
            config.OutputLayers = json["output_layers"].ToObject<List<NodeConfig>>();
            return config;
        }

        public static string to_snake_case(string name)
        {
            string intermediate = Regex.Replace(name, "(.)([A-Z][a-z0-9]+)", "$1_$2");
            string insecure = Regex.Replace(intermediate, "([a-z])([A-Z])", "$1_$2").ToLower();

            if (insecure[0] != '_')
            {
                return insecure;
            }

            return "private" + insecure;
        }

        /// <summary>
        /// Determines whether config appears to be a valid layer config.
        /// </summary>
        /// <param name="config"></param>
        /// <returns></returns>
        public static bool validate_config(JObject config)
        {
            return !config.ContainsKey(_LAYER_UNDEFINED_CONFIG_KEY);
        }
    }
}
