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
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// A `Node` describes the connectivity between two layers.
    /// 
    /// Each time a layer is connected to some new input,
    /// a node is added to `layer._inbound_nodes`.
    /// Each time the output of a layer is used by another layer,
    /// a node is added to `layer._outbound_nodes`.
    /// </summary>
    public partial class Node : INode
    {
        NodeArgs args;

        public int[] node_indices;
        public int[] tensor_indices;
        public Tensors input_tensors => is_input ? Outputs : args.InputTensors;
        public Tensors Outputs => args.Outputs;
        public TensorShape[] input_shapes;
        public TensorShape[] output_shapes;
        public List<Tensor> KerasInputs { get; set; } = new List<Tensor>();
        ILayer _layer;
        public ILayer Layer => _layer;
        public bool is_input => args.InputTensors == null;
        public long[] FlatInputIds { get; set; }
        public long[] FlatOutputIds { get; set; }
        bool _single_positional_tensor_passed => KerasInputs.Count() == 1;
        Dictionary<int, long> _keras_inputs_ids_and_indices = new Dictionary<int, long>();
        public INode[] ParentNodes
        {
            get
            {
                var node_deps = new List<INode>();
                foreach (var kt in KerasInputs)
                {
                    var (layer, node_index, _) = kt.KerasHistory;
                    if (layer != null)
                        node_deps.append(layer.InboundNodes[node_index]);
                }
                return node_deps.ToArray();
            }
        }

        public Node(NodeArgs args)
        {
            this.args = args;
        }

        public void Connect(Layer layer)
        {
            _layer = layer;

            if (args.InputTensors != null)
                KerasInputs.AddRange(args.InputTensors);
            
            foreach (var (i, ele) in enumerate(KerasInputs))
                _keras_inputs_ids_and_indices[i] = ele.Id;

            // Wire up Node to Layers.
            layer.InboundNodes.Add(this);
            
            foreach (var kt in KerasInputs)
            {
                if (kt.KerasHistory == null)
                    continue;
                var (inbound_layer, _, _) = kt.KerasHistory;
                if (inbound_layer != null)
                    inbound_layer.OutboundNodes.Add(this);
            }

            // Set metadata on outputs.
            var node_index = layer.InboundNodes.Count - 1;
            foreach (var (i, tensor) in enumerate(Outputs))
                tensor.KerasHistory = new KerasHistory(layer, node_index, i);

            // Cached for performance.
            FlatInputIds = KerasInputs.Select(x => x.Id).ToArray();
            FlatOutputIds = Outputs.Select(x => x.Id).ToArray();
        }

        /// <summary>
        /// Maps Keras Tensors to computed Tensors using `tensor_dict`.
        /// </summary>
        /// <param name="tensor_dict"></param>
        /// <returns></returns>
        public Tensors MapArguments(Dictionary<long, Queue<Tensor>> tensor_dict)
        {
            if (_single_positional_tensor_passed)
            {
                var kt_id = _keras_inputs_ids_and_indices[0];
                return tensor_dict[kt_id].Dequeue();
            }
            else
            {
                var flat_arguments = KerasInputs.Select(x => x).ToArray();
                foreach (var (kt_index, kt_id) in enumerate(_keras_inputs_ids_and_indices))
                    flat_arguments[kt_index] = tensor_dict[kt_id].Dequeue();

                return flat_arguments;
            }
        }

        public override string ToString()
            => $"{Layer.Name}, {KerasInputs.Count} inputs: {string.Join(",", KerasInputs.Select(x => x.name))}";
    }
}
