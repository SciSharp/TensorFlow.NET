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
    public abstract partial class Layer : AutoTrackable
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
        public List<IVariableV1> trainable_variables
        { 
            get 
            {
                if(trainableWeights.Count == 0)
                    _layers.ForEach(x => trainableWeights.AddRange(x.trainableWeights));

                return trainableWeights;
            } 
        } 

        protected List<IVariableV1> nonTrainableWeights;
        public List<IVariableV1> non_trainable_variables => nonTrainableWeights;

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
        /// <param name="state"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
        public Tensors Apply(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            Tensors outputs = null;

            callContext = callContext ?? new ThreadLocal<CallContext>()
            {
                Value = new CallContext()
            };

            var eager = tf.executing_eagerly();
            using var ctxManager = CallContext.enter();

            string nameScope = "";
            if (eager)
                nameScope = name;
            else
                nameScope = _name_scope();

            // using var graph = tf.keras.backend.get_graph().as_default();
            if (!inputs.IsEagerTensor)
                tf.Context.graph_mode();

            tf_with(ops.name_scope(nameScope), scope =>
            {
                if (!built)
                    MaybeBuild(inputs);

                outputs = call(inputs, state: state, is_training: is_training);

                outputs = _set_connectivity_metadata_(inputs, outputs);
                _handle_activity_regularization(inputs, outputs);
                _set_mask_metadata(inputs, outputs, null);
            });

            if (!inputs.IsEagerTensor)
                tf.Context.restore_mode();

            return outputs;
        }

        private Tensors _set_connectivity_metadata_(Tensors inputs, Tensors outputs)
        {
            /*var returnOutputs = new List<Tensor>();
            foreach(var x in outputs)
            {
                if (inputs.Contains(x))
                {

                }
                returnOutputs.Add(x);
            }*/

            new Node(this, new NodeArgs
            {
                Outputs = outputs
            });

            //_add_inbound_node(input_tensors: inputs, output_tensors: outputs);
            return outputs;
        }

        private void _handle_activity_regularization(Tensors inputs, Tensors outputs)
        {
            //if(_activity_regularizer != null)
            {

            }
        }

        private void _set_mask_metadata(Tensors inputs, Tensors outputs, Tensors previous_mask)
        {

        }

        private Tensor compute_mask(Tensor inputs, Tensor mask = null)
        {
            return null;
        }

        protected virtual Tensors call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            throw new NotImplementedException("");
        }

        protected virtual string _name_scope()
        {
            return Name;
        }

        protected void MaybeBuild(Tensors inputs)
        {
            // Check input assumptions set before layer building, e.g. input rank.
            if (built)
                return;
            if (DType == TF_DataType.DtInvalid)
                args.DType = inputs.dtype;

            var input_shapes = inputs.shape;
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
