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

using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public class Sequential : Model, ITensorFlowObject
    {
#pragma warning disable CS0649 // Field 'Sequential._is_graph_network' is never assigned to, and will always have its default value false
        bool _is_graph_network;
#pragma warning restore CS0649 // Field 'Sequential._is_graph_network' is never assigned to, and will always have its default value false
#pragma warning disable CS0169 // The field 'Sequential.outputs' is never used
        Tensor[] outputs;
#pragma warning restore CS0169 // The field 'Sequential.outputs' is never used

        bool computeOutputAndMaskJointly;
        bool autoTrackSubLayers;
        TensorShape inferredInputShape;
        bool hasExplicitInputShape;
        TF_DataType inputDType;
        Layer[] layers;

        public Sequential(Layer[] layers = null, string name = null) 
            : base(new ModelArgs { Name = name})
        {
            this.layers = layers ?? new Layer[0];
            SupportsMasking = true;
            computeOutputAndMaskJointly = true;
            autoTrackSubLayers = false;
            hasExplicitInputShape = false;
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
            if(layers.Length == 0)
            {
                if(layer is InputLayer)
                {
                    set_inputs = true;
                }
                else
                {
                    if (layer.BatchInputShape != null)
                    {
                        // Instantiate an input layer.
                        var x = tf.keras.Input(
                              batch_shape: layer.BatchInputShape,
                              dtype: layer.DType,
                              name: layer.Name + "_input");

                        // This will build the current layer
                        // and create the node connecting the current layer
                        // to the input layer we just created.
                        layer.Apply(x);
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
