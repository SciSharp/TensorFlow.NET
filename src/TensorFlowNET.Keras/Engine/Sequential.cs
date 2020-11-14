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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Sequential` groups a linear stack of layers into a `tf.keras.Model`.
    /// `Sequential` provides training and inference features on this model.
    /// </summary>
    public class Sequential : Model
    {
        SequentialArgs args;
        bool _is_graph_network;
        Tensor inputs;
        Tensor outputs;

        bool computeOutputAndMaskJointly;
        bool autoTrackSubLayers;
        TensorShape inferredInputShape;
        bool hasExplicitInputShape;
        TF_DataType inputDType;
        List<ILayer> layers => args.Layers;
        public TensorShape output_shape => outputs.TensorShape;
        bool built = false;

        public Sequential(SequentialArgs args)
            : base(new ModelArgs
            {
                Name = args.Name
            })
        {
            this.args = args;
            if (args.Layers == null)
                args.Layers = new List<ILayer>();
            // SupportsMasking = true;
            computeOutputAndMaskJointly = true;
            autoTrackSubLayers = false;
            hasExplicitInputShape = false;
            _is_graph_network = false;
        }

        public void add(Tensor tensor)
        {
            var layer = tensor.KerasHistory.Layer as Layer;
            add(layer);
        }

        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// <param name="layer"></param>
        public void add(Layer layer)
        {
            built = false;
            var set_inputs = false;
            if (layers.Count == 0)
            {
                if (layer is InputLayer)
                {
                    set_inputs = true;
                }
                else
                {
                    if (layer.BatchInputShape != null)
                    {
                        // Instantiate an input layer.
                        var x = keras.Input(
                              shape: layer.BatchInputShape,
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
                    outputs = layer.InboundNodes[^1].Outputs;
                }

            }
            else if (outputs != null)
            {
                outputs = layer.Apply(outputs);
            }

            if (set_inputs || _is_graph_network)
            {
                _init_graph_network(inputs, outputs);
            }
            else
            {

            }
        }

        void _init_graph_network(Tensor inputs, Tensor outputs)
        {
            _is_graph_network = true;
            this.inputs = inputs;
            this.outputs = outputs;
            built = true;
            _map_graph_network(inputs, outputs);
        }

        void _map_graph_network(Tensor inputs, Tensor outputs)
        {
            layers.add(outputs.KerasHistory.Layer);
        }
    }
}
