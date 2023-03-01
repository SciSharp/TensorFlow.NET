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
using System.Diagnostics;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Saving;
using Tensorflow.Train;

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
            return class_name switch
            {
                "Sequential" => new Sequential(config.ToObject<SequentialArgs>()),
                "InputLayer" => new InputLayer(config.ToObject<InputLayerArgs>()),
                "Flatten" => new Flatten(config.ToObject<FlattenArgs>()),
                "ELU" => new ELU(config.ToObject<ELUArgs>()),
                "Dense" => new Dense(config.ToObject<DenseArgs>()),
                "Softmax" => new Softmax(config.ToObject<SoftmaxArgs>()),
                _ => throw new NotImplementedException($"The deserialization of <{class_name}> has not been supported. Usually it's a miss during the development. " +
                        $"Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues")
            };
        }

        public static Layer deserialize_keras_object(string class_name, LayerArgs args)
        {
            return class_name switch
            {
                "Sequential" => new Sequential(args as SequentialArgs),
                "InputLayer" => new InputLayer(args as InputLayerArgs),
                "Flatten" => new Flatten(args as FlattenArgs),
                "ELU" => new ELU(args as ELUArgs),
                "Dense" => new Dense(args as DenseArgs),
                "Softmax" => new Softmax(args as SoftmaxArgs),
                _ => throw new NotImplementedException($"The deserialization of <{class_name}> has not been supported. Usually it's a miss during the development. " +
                        $"Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues")
            };
        }

        public static LayerArgs? deserialize_layer_args(string class_name, JToken config)
        {
            return class_name switch
            {
                "Sequential" => config.ToObject<SequentialArgs>(),
                "InputLayer" => config.ToObject<InputLayerArgs>(),
                "Flatten" => config.ToObject<FlattenArgs>(),
                "ELU" => config.ToObject<ELUArgs>(),
                "Dense" => config.ToObject<DenseArgs>(),
                "Softmax" => config.ToObject<SoftmaxArgs>(),
                _ => throw new NotImplementedException($"The deserialization of <{class_name}> has not been supported. Usually it's a miss during the development. " +
                        $"Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues")
            };
        }

        public static ModelConfig deserialize_model_config(JToken json)
        {
            ModelConfig config = new ModelConfig();
            config.Name = json["name"].ToObject<string>();
            config.Layers = new List<LayerConfig>();
            var layersToken = json["layers"];
            foreach (var token in layersToken)
            {
                var args = deserialize_layer_args(token["class_name"].ToObject<string>(), token["config"]);
                config.Layers.Add(new LayerConfig()
                {
                    Config = args, 
                    Name = token["name"].ToObject<string>(), 
                    ClassName = token["class_name"].ToObject<string>(), 
                    InboundNodes = token["inbound_nodes"].ToObject<List<NodeConfig>>()
                });
            }
            config.InputLayers = json["input_layers"].ToObject<List<NodeConfig>>();
            config.OutputLayers = json["output_layers"].ToObject<List<NodeConfig>>();
            return config;
        }

        public static string to_snake_case(string name)
        {
            return string.Concat(name.Select((x, i) =>
            {
                return i > 0 && char.IsUpper(x) && !Char.IsDigit(name[i - 1]) ?
                    "_" + x.ToString() :
                    x.ToString();
            })).ToLower();
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
