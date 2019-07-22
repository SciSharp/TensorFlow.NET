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

using System.Collections.Generic;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Engine
{
    public class Network : Layer
    {
        protected bool _is_compiled;
        protected bool _expects_training_arg;
        protected bool _compute_output_and_mask_jointly;
        /// <summary>
        /// All layers in order of horizontal graph traversal.
        /// Entries are unique. Includes input and output layers.
        /// </summary>
        protected List<Layer> _layers;

        public Network(string name = null) 
            : base(name: name)
        {
            _init_subclassed_network(name);
        }

        protected virtual void _init_subclassed_network(string name = null)
        {
            _base_init(name: name);
        }

        protected virtual void _base_init(string name = null)
        {
            _init_set_name(name);
            trainable = true;
            _is_compiled = false;
            _expects_training_arg = false;
            _compute_output_and_mask_jointly = false;
            supports_masking = false;
            _layers = new List<Layer>();
        }
    }
}
