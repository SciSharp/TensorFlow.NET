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
        bool dynamic = true;
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

        protected string name;
        protected string base_name;
        public string Name => name;
        
        protected bool computePreviousMask;
        protected List<Operation> updates;
        public TensorShape BatchInputShape => args.BatchInputShape;

        List<Node> inboundNodes;
        public List<Node> InboundNodes => inboundNodes;

        List<Node> outboundNodes;
        public List<Node> OutboundNodes => outboundNodes;

        ThreadLocal<CallContext> callContext;
        public CallContext CallContext => callContext.Value;
        public static List<KerasHistory> KerasHistories = new List<KerasHistory>();

        public Layer(LayerArgs args)
        {
            this.args = args;
            // A stateful layer is a layer whose updates are run during inference too,
            // for instance stateful RNNs.
            stateful = false;
            // Indicates whether `build` needs to be called upon layer call, to create
            // the layer's weights.
            built = false;
            SupportsMasking = false;

            _init_set_name(args.Name);
            trainableWeights = new List<IVariableV1>();
            nonTrainableWeights = new List<IVariableV1>();
            computePreviousMask = false;
            updates = new List<Operation>();

            inboundNodes = new List<Node>();
            outboundNodes = new List<Node>();

            // Manage input shape information if passed.
            if (args.BatchInputShape == null && args.InputShape != null)
            {
                args.BatchInputShape = new int[] { args.BatchSize }.Concat(args.InputShape.dims).ToArray();
            }
        }

        bool _in_functional_construction_mode(Tensors inputs)
        {
            return tf.Context.executing_eagerly() 
                && inputs.Count(x => !x.IsEagerTensor) == inputs.Count();
        }

        public void SetConnectivityMetadata(Tensors inputs, Tensors outputs)
        {

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
                InputTensors = inputs,
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

        /// <summary>
        /// Subclass has to override this method.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="state"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
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

            tf.init_scope();

            tf.Context.eager_mode();
            build(inputs.shape);
            tf.Context.restore_mode();
            
            built = true;
        }

        protected virtual void build(TensorShape input_shape)
        {
            built = true;
        }

        protected virtual void add_loss(Func<Tensor> losses)
        {
            
        }

        /// <summary>
        /// Create lambdas which compute regularization losses.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="variable"></param>
        /// <param name="regularizer"></param>
        void _handle_weight_regularization(string name, IVariableV1 variable, IRegularizer regularizer)
        {
            add_loss(() => regularizer.Apply(new RegularizerArgs
            {
                
            }));
        }

        protected virtual void add_update(Tensor[] updates, bool inputs = false)
        {
            var updates_op = updates.Select(x => x.op).ToArray();
            this.updates.AddRange(updates_op);
        }

        // Determine layer name (non-unique).
        protected virtual void _init_set_name(string name, bool zero_based = true)
        {
            base_name = name;
            this.name = name;
            if (name == null)
            {
                base_name = generic_utils.to_snake_case(this.GetType().Name);
                this.name = base_layer_utils.unique_layer_name(base_name, zero_based: zero_based);
            }
        }
    }
}
