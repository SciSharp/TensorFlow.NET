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
using System.Threading;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Base layer class.
    /// A layer is a class implementing common neural networks operations, such
    /// as convolution, batch norm, etc. These operations require managing weights,
    /// losses, updates, and inter-layer connectivity.
    /// </summary>
    public abstract class Layer : AutoTrackable
    {
        /// <summary>
        /// Arguments initialize layer.
        /// </summary>
        LayerArgs args;

        /// <summary>
        /// Indicates whether `build` needs to be called upon layer call, to create
        /// the layer's weights.
        /// </summary>
        protected bool built;
        public bool Trainable => args.Trainable;
        public TF_DataType DType => args.DType;

        /// <summary>
        /// A stateful layer is a layer whose updates are run during inference too,
        /// for instance stateful RNNs.
        /// </summary>
        protected bool stateful;
        /// <summary>
        /// Provides information about which inputs are compatible with the layer.
        /// </summary>
        protected InputSpec inputSpec;
        public bool SupportsMasking { get; set; }
        protected List<IVariableV1> trainableWeights;
        public List<IVariableV1> TrainableVariables => trainableWeights;
        protected List<IVariableV1> nonTrainableWeights;

        string name;
        public string Name => name;

        protected string baseName;
        protected bool computePreviousMask;
        protected List<Operation> updates;
        public TensorShape BatchInputShape => args.BatchInputShape;

        List<Node> inboundNodes;
        public List<Node> InboundNodes => inboundNodes;

        List<Node> outboundNodes;
        public List<Node> OutboundNodes => outboundNodes;

        ThreadLocal<CallContext> callContext;
        public CallContext CallContext => callContext.Value;

        public Layer(LayerArgs args)
        {
            this.args = args;
            // A stateful layer is a layer whose updates are run during inference too,
            // for instance stateful RNNs.
            stateful = false;
            // Indicates whether `build` needs to be called upon layer call, to create
            // the layer's weights.
            built = false;
            this.SupportsMasking = false;

            _init_set_name(name);
            trainableWeights = new List<IVariableV1>();
            nonTrainableWeights = new List<IVariableV1>();
            computePreviousMask = false;
            updates = new List<Operation>();

            inboundNodes = new List<Node>();

            // Manage input shape information if passed.
            if(args.BatchInputShape == null && args.InputShape != null)
            {
                args.BatchInputShape = new int[] { args.BatchSize }.Concat(args.InputShape.dims).ToArray();
            }
        }

        /// <summary>
        /// Wraps `call`, applying pre- and post-processing steps.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
        public Tensor Apply(Tensor[] inputs, bool is_training = false)
        {
            callContext = callContext ?? new ThreadLocal<CallContext>()
            {
                Value = new CallContext()
            };

            using var ctxManager = CallContext.enter();

            string nameScope = "";
            if (tf.Context.executing_eagerly())
            {
                nameScope = name;
            }
            else
            {
                throw new NotImplementedException("");
            }

            tf_with(ops.name_scope(nameScope), scope =>
            {
                if (!built)
                    MaybeBuild(inputs);

                call(inputs, is_training: is_training);
            });

            throw new NotImplementedException("");
        }

        [Obsolete("User Apply()")]
        public Tensor[] __call__(Tensor[] inputs,
            Tensor training = null,
            Tensor state = null,
            VariableScope scope = null)
        {
            var input_list = inputs;
            var input = inputs[0];
            Tensor[] outputs = null;

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
                var graph = tf.keras.backend.get_graph().as_default();
                tf_with(ops.name_scope(_name_scope()), delegate
                {
                    // Build layer if applicable (if the `build` method has been
                    // overridden).
                    MaybeBuild(inputs);

                    outputs = call(inputs, 
                        // training: training,
                        state: state);

                    (input, outputs) = _set_connectivity_metadata_(input, outputs);
                    _handle_activity_regularization(inputs[0], outputs);
                    _set_mask_metadata(inputs[0], outputs, null);
                });
            }

            return outputs;
        }

        private (Tensor, Tensor[]) _set_connectivity_metadata_(Tensor inputs, Tensor[] outputs)
        {
            //_add_inbound_node(input_tensors: inputs, output_tensors: outputs);
            return (inputs, outputs);
        }

        private void _handle_activity_regularization(Tensor inputs, Tensor[] outputs)
        {
            //if(_activity_regularizer != null)
            {

            }
        }

        private void _set_mask_metadata(Tensor inputs, Tensor[] outputs, Tensor previous_mask)
        {

        }

        private Tensor compute_mask(Tensor inputs, Tensor mask = null)
        {
            return null;
        }

        protected virtual Tensor[] call(Tensor[] inputs, bool is_training = false, Tensor state = null)
        {
            throw new NotImplementedException("");
        }

        protected virtual string _name_scope()
        {
            return Name;
        }

        protected void MaybeBuild(Tensor[] inputs)
        {
            // Check input assumptions set before layer building, e.g. input rank.
            if (built)
                return;
            if (DType == TF_DataType.DtInvalid)
                args.DType = inputs[0].dtype;

            var input_shapes = inputs[0].TensorShape;
            build(input_shapes);
            built = true;
        }

        protected virtual void build(TensorShape input_shape)
        {
            built = true;
        }

        protected virtual IVariableV1 add_weight(string name,
            TensorShape shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null,
            Func<VariableArgs, IVariableV1> getter = null)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            if (trainable == null)
                trainable = true;

            // Initialize variable when no initializer provided
            if (initializer == null)
            {
                // If dtype is DT_FLOAT, provide a uniform unit scaling initializer
                if (dtype.is_floating())
                    initializer = tf.glorot_uniform_initializer;
                else if (dtype.is_integer())
                    initializer = tf.zeros_initializer;
                else
                    throw new ValueError($"An initializer for variable {name} of type {dtype.as_base_dtype()} is required for layer {this.Name}");
            }

            var args = new VariableArgs
            {
                Name = name,
                Shape = shape,
                DType = dtype,
                Getter = getter ?? base_layer_utils.make_variable,
                Overwrite = true,
                Initializer = initializer,
                Trainable = trainable.Value
            };
            var variable = _add_variable_with_custom_getter(args);

            //backend.track_variable(variable);
            if (trainable == true)
                trainableWeights.Add(variable);
            else
                nonTrainableWeights.Add(variable);

            return variable;
        }

        protected virtual void add_update(Tensor[] updates, bool inputs = false)
        {
            var updates_op = updates.Select(x => x.op).ToArray();
            this.updates.AddRange(updates_op);
        }

        // Determine layer name (non-unique).
        protected virtual void _init_set_name(string name, bool zero_based = true)
        {
            var base_name = name;
            this.name = name;
            if (name == null)
                (this.name, baseName) = _make_unique_name();
        }

        protected virtual (string, string) _make_unique_name()
        {
            string base_name = generic_utils.to_snake_case(this.GetType().Name);
            string name = base_layer_utils.unique_layer_name(base_name);
            return (name, base_name);
        }
    }
}
