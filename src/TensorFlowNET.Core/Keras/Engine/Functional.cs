using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// A `Functional` model is a `Model` defined as a directed graph of layers.
    /// </summary>
    public class Functional : Model
    {
        TensorShape _build_input_shape;
        bool _compute_output_and_mask_jointly;
        bool _expects_training_arg;
        bool _expects_mask_arg;
        bool _autocast;
        List<Layer> _output_layers;
        List<Layer> _input_layers;
        List<KerasHistory> _input_coordinates;
        List<KerasHistory> _output_coordinates;

        public Functional(Tensors inputs, Tensors outputs) 
            : base(new ModelArgs
            {
                Inputs = inputs,
                Outputs = outputs
            })
        {
            _input_layers = new List<Layer>();
            _output_layers = new List<Layer>();
            _input_coordinates = new List<KerasHistory>();
            _output_coordinates = new List<KerasHistory>();
            _init_graph_network(inputs, outputs);
        }

        void _init_graph_network(Tensors inputs, Tensors outputs)
        {
            _is_graph_network = true;
            this.inputs = inputs;
            this.outputs = outputs;
            built = true;
            _build_input_shape = inputs.shape;
            _compute_output_and_mask_jointly = true;
            _expects_training_arg = true;
            _expects_mask_arg = true;
            // A graph network does not autocast inputs, as its layers will cast them instead.
            _autocast = false;

            if (outputs.Any(x => x.KerasHistory == null))
                base_layer_utils.create_keras_history(outputs);

            // Build self._output_layers:
            foreach (var x in outputs)
            {
                var (layer, node_index, tensor_index) = x.KerasHistory;
                _output_layers.append(layer);
                _output_coordinates.append(new KerasHistory(layer, node_index, tensor_index, x));
            }

            // Build self._input_layers:
            foreach(var x in inputs)
            {
                var (layer, node_index, tensor_index) = x.KerasHistory;
                _input_layers.append(layer);
                _input_coordinates.append(new KerasHistory(layer, node_index, tensor_index, x));
            }
        }
       
    }
}
