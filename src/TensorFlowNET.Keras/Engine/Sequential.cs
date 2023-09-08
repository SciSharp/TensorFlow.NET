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
using System.Linq;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using static Tensorflow.KerasApi;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Sequential` groups a linear stack of layers into a `tf.keras.Model`.
    /// `Sequential` provides training and inference features on this model.
    /// </summary>
    public class Sequential : Functional
    {
        SequentialArgs args;

        bool _compute_output_and_mask_jointly;
        bool _auto_track_sub_layers;
        Shape _inferred_input_shape;
        bool _has_explicit_input_shape;
        bool _graph_initialized;
        public Shape output_shape => outputs[0].shape;
        List<INode> _created_nodes;

        public Sequential(SequentialArgs args)
            : base(args.Inputs, args.Outputs, name: args.Name)
        {
            this.args = args;
            // SupportsMasking = true;
            _compute_output_and_mask_jointly = true;
            _auto_track_sub_layers = false;
            _has_explicit_input_shape = false;
            _is_graph_network = false;
            _created_nodes = new List<INode>();

            // Add to the model any layers passed to the constructor.
            if (args.Layers is not null)
            {
                InitLayers(args.Layers);
            }
        }

        public void InitLayers(IEnumerable<ILayer> layers)
        {
            foreach(var layer in layers)
            {
                // TODO(Rinne): remove it and completely fix issue 1084
                if(layer is Sequential s)
                {
                    s.Layers.ForEach(x => ((Layer)x).enforce_layer_construction());
                }
                add(layer);
                // TODO(Rinne): remove it and completely fix issue 1084
                if (layer is Sequential s2)
                {
                    s2.Layers.ForEach(x => ((Layer)x).unset_layer_construction());
                }
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
            if (_self_tracked_trackables.Count == 0)
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
                              batch_input_shape: layer.BatchInputShape.ToSingleShape(),
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
                    outputs = layer.InboundNodes.Last().Outputs;
                    inputs = layer_utils.get_source_inputs(outputs[0]);
                    built = true;
                    _has_explicit_input_shape = true;
                }
            }
            else if (outputs != null)
            {
                // If the model is being built continuously on top of an input layer:
                // refresh its output.
                outputs = layer.Apply(outputs);
                built = true;
            }

            if (set_inputs || _is_graph_network)
            {
                _init_graph_network(inputs, outputs);
                _graph_initialized = true;
            }
            else
            {
                _self_tracked_trackables.add(layer);
                // TODO(Rinne): self._handle_deferred_layer_dependencies([layer])
            }
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            if (!_has_explicit_input_shape)
            {
                _build_graph_network_for_inferred_shape(inputs.shape, inputs.dtype);
            }

            if(_graph_initialized)
            {
                if (!built)
                    _init_graph_network(this.inputs, outputs);
                return base.Call(inputs, state, training);
            }

            return base.Call(inputs, state, training);
        }

        void _build_graph_network_for_inferred_shape(Shape input_shape, TF_DataType input_dtype)
        {
            if (_inferred_input_shape == input_shape)
                return;

            ops.init_scope();
            var inputs = keras.Input(batch_input_shape: input_shape,
                dtype: input_dtype,
                name: _self_tracked_trackables[0].Name.EndsWith("_input") ? _self_tracked_trackables[0].Name : $"{_self_tracked_trackables[0].Name}_input");
            Tensors layer_input = inputs;
            Tensors layer_output = null;
            Tensors outputs = null;
            List<INode> created_nodes = new List<INode>();
            foreach (var layer in Layers)
            {
                clear_previously_created_nodes(layer, _created_nodes);
                layer_output = layer.Apply(layer_input);
                // Keep track of nodes just created above
                track_nodes_created_by_last_call(layer, created_nodes);
                layer_input = layer_output;
                outputs = layer_output;
            }
            _created_nodes = created_nodes;
            _init_graph_network(inputs, outputs);
            _graph_initialized = true;
            _inferred_input_shape = input_shape;
        }

        void clear_previously_created_nodes(ILayer layer, List<INode> created_nodes)
        {
            foreach(var node in layer.InboundNodes)
            {
                foreach(var prev_layer in node.InboundLayers)
                {
                    var outNodes = prev_layer.OutboundNodes.Where(x => !created_nodes.Contains(x)).ToArray();
                    prev_layer.OutboundNodes.Clear();
                    prev_layer.OutboundNodes.AddRange(outNodes);
                }
            }

            var inNodes = layer.InboundNodes.Where(x => !created_nodes.Contains(x)).ToArray();
            layer.InboundNodes.Clear();
            layer.InboundNodes.AddRange(inNodes);
        }

        void track_nodes_created_by_last_call(ILayer layer, List<INode> created_nodes)
        {
            var node = layer.InboundNodes.Last();
            created_nodes.Add(node);
            foreach(var prev_layer in node.InboundLayers)
            {
                created_nodes.add(prev_layer.OutboundNodes.Last());
            }
        }

        public override List<ILayer> Layers
            => base.Layers.Where(x => x is not InputLayer).ToList();
    }
}
