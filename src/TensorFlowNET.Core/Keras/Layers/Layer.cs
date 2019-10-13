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
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Base layer class.
    /// A layer is a class implementing common neural networks operations, such
    /// as convolution, batch norm, etc. These operations require managing weights,
    /// losses, updates, and inter-layer connectivity.
    /// 
    /// tensorflow\python\keras\engine\base_layer.py
    /// </summary>
    public class Layer : AutoTrackable
    {
        /// <summary>
        /// Indicates whether `build` needs to be called upon layer call, to create
        /// the layer's weights.
        /// </summary>
        protected bool built;
        protected bool trainable;
        public TF_DataType _dtype;
        /// <summary>
        /// A stateful layer is a layer whose updates are run during inference too,
        /// for instance stateful RNNs.
        /// </summary>
        protected bool stateful;
        /// <summary>
        /// Provides information about which inputs are compatible with the layer.
        /// </summary>
        protected InputSpec input_spec;
        protected bool supports_masking;
        protected List<VariableV1> _trainable_weights;
        private string _name;
        public string name => _name;
        protected string _base_name;
        protected bool _compute_previous_mask;
        protected List<Operation> _updates;
        public int[] _batch_input_shape;

        private List<Node> _inbound_nodes;
        public List<Node> inbound_nodes => _inbound_nodes;

        private List<Node> _outbound_nodes;
        public List<Node> outbound_nodes => _outbound_nodes;

        float _initial_weights;

        public Layer(bool trainable = true, 
            string name = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] input_shape = null)
        {
            this.trainable = trainable;
            this._dtype = dtype;
            // A stateful layer is a layer whose updates are run during inference too,
            // for instance stateful RNNs.
            stateful = false;
            // Indicates whether `build` needs to be called upon layer call, to create
            // the layer's weights.
            built = false;
            this.supports_masking = false;

            _init_set_name(name);
            _trainable_weights = new List<VariableV1>();
            _compute_previous_mask = false;
            _updates = new List<Operation>();

            // Manage input shape information if passed.
            if(input_shape != null)
            {
                var shapes = new List<int> { -1 };
                shapes.AddRange(input_shape);
                _batch_input_shape = shapes.ToArray();
            }
            

            _dtype = dtype;

            _inbound_nodes = new List<Node>();
        }

        public Tensor __call__(Tensor[] inputs,
            Tensor training = null,
            VariableScope scope = null)
        {
            var input_list = inputs;
            var input = inputs[0];
            Tensor outputs = null;

            // We will attempt to build a TF graph if & only if all inputs are symbolic.
            // This is always the case in graph mode. It can also be the case in eager
            // mode when all inputs can be traced back to `keras.Input()` (when building
            // models using the functional API).
            bool build_graph = tf_utils.are_all_symbolic_tensors(input_list);

            if (build_graph)
            {
                // Only create Keras history if at least one tensor originates from a
                // `keras.Input`. Otherwise this Layer may be being used outside the Keras
                // framework.
                // base_layer_utils.create_keras_history(inputs)
            }

            // with base_layer_utils.call_context(self):

            // Handle Keras mask propagation from previous layer to current layer.
            // with base_layer_utils.call_context(self):
            // Check input assumptions set after layer building, e.g. input shape.
            if (build_graph)
            {
                // Symbolic execution on symbolic tensors. We will attempt to build
                // the corresponding TF subgraph inside `backend.get_graph()`
                var graph = backend.get_graph().as_default();
                tf_with(ops.name_scope(_name_scope()), delegate
                {
                    // Build layer if applicable (if the `build` method has been
                    // overridden).
                    _maybe_build(inputs[0]);

                    outputs = call(inputs[0], training: training);
                    (input, outputs) = _set_connectivity_metadata_(input, outputs);
                    _handle_activity_regularization(inputs[0], outputs);
                    _set_mask_metadata(inputs[0], outputs, null);
                });
            }

            return outputs;
        }

        private (Tensor, Tensor) _set_connectivity_metadata_(Tensor inputs, Tensor outputs)
        {
            //_add_inbound_node(input_tensors: inputs, output_tensors: outputs);
            return (inputs, outputs);
        }

        private void _handle_activity_regularization(Tensor inputs, Tensor outputs)
        {
            //if(_activity_regularizer != null)
            {

            }
        }

        private void _set_mask_metadata(Tensor inputs, Tensor outputs, Tensor previous_mask)
        {

        }

        private Tensor compute_mask(Tensor inputs, Tensor mask = null)
        {
            return null;
        }

        protected virtual Tensor call(Tensor inputs, Tensor training = null)
        {
            return inputs;
        }

        protected virtual string _name_scope()
        {
            return name;
        }

        protected void _maybe_build(Tensor input)
        {
            // Check input assumptions set before layer building, e.g. input rank.
            if (built)
                return;
            if (_dtype == TF_DataType.DtInvalid)
                _dtype = input.dtype;

            var input_shapes = input.TensorShape;
            build(input_shapes);
            built = true;
        }

        protected virtual void build(TensorShape input_shape)
        {
            built = true;
        }

        protected virtual VariableV1 add_weight(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null,
            Func<string, int[], TF_DataType, IInitializer, bool, VariableV1> getter = null)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            if (trainable == null)
                trainable = true;

            // Initialize variable when no initializer provided
            if(initializer == null)
            {
                // If dtype is DT_FLOAT, provide a uniform unit scaling initializer
                if (dtype.is_floating())
                    initializer = tf.glorot_uniform_initializer;
                else if (dtype.is_integer())
                    initializer = tf.zeros_initializer;
                else
                    throw new ValueError($"An initializer for variable {name} of type {dtype.as_base_dtype()} is required for layer {this.name}");
            }
            var variable = _add_variable_with_custom_getter(name,
                shape,
                dtype: dtype,
                getter: (getter == null) ? base_layer_utils.make_variable : getter,
                overwrite: true,
                initializer: initializer,
                trainable: trainable.Value);
            //backend.track_variable(variable);
            _trainable_weights.Add(variable);

            return variable;
        }

        protected virtual void add_update(Tensor[] updates, bool inputs = false)
        {
            var updates_op = updates.Select(x => x.op).ToArray();
            _updates.AddRange(updates_op);
        }

        // Determine layer name (non-unique).
        protected virtual void _init_set_name(string name, bool zero_based = true)
        {
            var base_name = name;
            _name = name;
            if (name == null)
                (_name, base_name) = _make_unique_name();
            _base_name = base_name;
        }

        protected virtual (string, string) _make_unique_name()
        {
            string base_name = generic_utils.to_snake_case(this.GetType().Name);
            string name = base_layer_utils.unique_layer_name(base_name);
            return (name, base_name);
        }
    }
}
