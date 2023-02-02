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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Utils
{
    public class generic_utils
    {
        /// <summary>
        /// This method does not have corresponding method in python. It's close to `serialize_keras_object`.
        /// </summary>
        /// <param name="instance"></param>
        /// <returns></returns>
        public static LayerConfig serialize_layer_to_config(ILayer instance)
        {
            var config = instance.get_config();
            return new LayerConfig
            {
                Config = config,
                ClassName = instance.GetType().Name
            };
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
    }
}
