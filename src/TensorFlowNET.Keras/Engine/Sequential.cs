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
using Tensorflow.Keras.Utils;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Sequential` groups a linear stack of layers into a `tf.keras.Model`.
    /// `Sequential` provides training and inference features on this model.
    /// </summary>
    public class Sequential : Functional
    {
        SequentialArgs args;
        bool _is_graph_network;
        Tensors inputs;
        Tensors outputs;

        bool _compute_output_and_mask_jointly;
        bool _auto_track_sub_layers;
        TensorShape _inferred_input_shape;
        bool _has_explicit_input_shape;
        TF_DataType _input_dtype;
        
        public TensorShape output_shape => outputs[0].TensorShape;
        bool built = false;

        public Sequential(SequentialArgs args)
            : base(args.Inputs, args.Outputs, name: args.Name)
        {
            this.args = args;
            if (args.Layers == null)
                args.Layers = new List<ILayer>();
            // SupportsMasking = true;
            _compute_output_and_mask_jointly = true;
            _auto_track_sub_layers = false;
            _has_explicit_input_shape = false;
            _is_graph_network = false;

            // Add to the model any layers passed to the constructor.
            if (args.Layers != null)
            {
                foreach (var layer in args.Layers)
                    add(layer as Layer);
            }
        }

        public void add(Tensor tensor)
        {
            var layer = tensor.KerasHistory.Layer;
            add(layer);
        }

        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// <param name="layer"></param>
        public void add(ILayer layer)
        {
            built = false;
            var set_inputs = false;
            if (_layers.Count == 0)
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
                              batch_input_shape: layer.BatchInputShape,
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
                    inputs = layer_utils.get_source_inputs(outputs[0]);
                    built = true;
                    _has_explicit_input_shape = true;
                }
            }
            else if (outputs != null)
            {
                outputs = layer.Apply(outputs);
                built = true;
            }

            if (set_inputs || _is_graph_network)
            {
                _init_graph_network(inputs, outputs);
                _is_graph_network = true;
            }
            else
            {

            }
        }
    }
}
