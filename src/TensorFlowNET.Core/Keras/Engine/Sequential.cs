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

using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Engine
{
    public class Sequential : Model, IObjectLife
    {
        bool _is_graph_network;
        Tensor[] outputs;

        public Sequential(string name = null) 
            : base(name: name)
        {
            supports_masking = true;
            _compute_output_and_mask_jointly = true;
        }

        public void __enter__()
        {
            
        }

        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// <param name="layer"></param>
        public void add(Layer layer)
        {
            built = false;
            var set_inputs = false;
            if(_layers.Count == 0)
            {
                if(layer is InputLayer)
                {

                }
                else
                {
                    var (batch_shape, dtype) = (layer._batch_input_shape, layer._dtype);
                    if (batch_shape != null)
                    {
                        // Instantiate an input layer.
                        var x = keras.layers.Input(
                              batch_shape: batch_shape,
                              dtype: dtype,
                              name: layer.name + "_input");

                        // This will build the current layer
                        // and create the node connecting the current layer
                        // to the input layer we just created.
                        layer.__call__(x);
                        set_inputs = true;
                    }
                }

                if (set_inputs)
                {
                    // If an input layer (placeholder) is available.
                    // outputs = layer.inbound_nodes;
                }

            }

            if (set_inputs || _is_graph_network)
            {

            }
        }

        public void __exit__()
        {
            
        }

        public void Dispose()
        {

        }

        public void __init__()
        {
            
        }

        public void __del__()
        {
            
        }
    }
}
