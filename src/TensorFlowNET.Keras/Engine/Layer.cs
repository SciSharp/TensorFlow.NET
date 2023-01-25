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
using Tensorflow.Eager;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
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
    public abstract partial class Layer : AutoTrackable, ILayer
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
        public bool Built => built;
        public bool Trainable => args.Trainable;
        public TF_DataType DType => args.DType;
        public bool AutoCast => args.Autocast;
        public IRegularizer ActivityRegularizer => args.ActivityRegularizer;

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
        protected List<IVariableV1> _trainable_weights;

        public virtual List<IVariableV1> TrainableVariables => _trainable_weights;

        protected List<IVariableV1> _non_trainable_weights;
        public List<IVariableV1> non_trainable_variables => _non_trainable_weights;

        protected int id;
        public int Id => id;
        protected string name;
        protected string base_name;
        public string Name => name;

        protected bool computePreviousMask;
        protected List<Operation> updates;
        public Shape BatchInputShape => args.BatchInputShape;

        List<INode> inboundNodes;
        public List<INode> InboundNodes => inboundNodes;

        List<INode> outboundNodes;
        public List<INode> OutboundNodes => outboundNodes;

        ThreadLocal<CallContext> callContext = new ThreadLocal<CallContext>();
        public CallContext CallContext => callContext.Value;
        public Tensor[] input => inboundNodes[0].input_tensors;
        public Dictionary<int, List<INode>> NodesByDepth { get; set; }
        public Shape OutputShape => inboundNodes[0].Outputs.shape;
        protected List<ILayer> _self_tracked_trackables;

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

            id = ops.uid_layer();
            _init_set_name(args.Name);
            _trainable_weights = new List<IVariableV1>();
            _non_trainable_weights = new List<IVariableV1>();
            computePreviousMask = false;
            updates = new List<Operation>();
            _self_tracked_trackables = new List<ILayer>();

            inboundNodes = new List<INode>();
            outboundNodes = new List<INode>();

            // Manage input shape information if passed.
            if (args.BatchInputShape == null && args.InputShape != null)
            {
                args.BatchInputShape = new long[] { args.BatchSize }.Concat(args.InputShape.dims).ToArray();
            }
        }

        bool _in_functional_construction_mode(Tensors inputs)
        {
            return tf.Context.executing_eagerly()
                && inputs.Count(x => x is not EagerTensor && x is not NDArray) == inputs.Count();
        }

        public void SetConnectivityMetadata(Tensors inputs, Tensors outputs)
            => _set_connectivity_metadata_(inputs, outputs);

        private void _set_connectivity_metadata_(Tensors inputs, Tensors outputs)
        {
            var node = new Node(new NodeArgs
            {
                InputTensors = inputs,
                Outputs = outputs
            });
            node.Connect(this);
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
        /// <param name="training"></param>
        /// <returns></returns>
        protected virtual Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            return inputs;
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

            bool need_restore_mode = false;
            if (inputs.Any(x => x is EagerTensor) || tf.Context.is_build_function())
            {
                need_restore_mode = true;
                tf.Context.eager_mode(isFunc: tf.Context.is_build_function());
            }
               
            build(inputs.shape);

            if (need_restore_mode)
                tf.Context.restore_mode();

            built = true;
        }

        public virtual void build(Shape input_shape)
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

            add_loss(() => tf_with(ops.name_scope(name + "/Regularizer"), scope => 
                regularizer.Apply(new RegularizerArgs(variable.AsTensor())
                {

                }) 
            ));
        }

        /*protected virtual void add_update(Tensor[] updates, bool inputs = false)
        {
            var updates_op = updates.Select(x => x.op).ToArray();
            this.updates.AddRange(updates_op);
        }*/

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

        public int count_params()
        {
            if (Trainable)
                return layer_utils.count_params(this, weights);
            return 0;
        }
        List<IVariableV1> ILayer.TrainableWeights
        {
            get
            {
                return _trainable_weights;
            }
        }

        List<IVariableV1> ILayer.NonTrainableWeights
        {
            get
            {
                return _non_trainable_weights;
            }
        }

        public List<IVariableV1> weights
        {
            get
            {
                var weights = new List<IVariableV1>();
                weights.AddRange(_trainable_weights);
                weights.AddRange(_non_trainable_weights);
                return weights;
            }
            set
            {
                if (weights.Count() != value.Count()) throw new ValueError(
                                            $"You called `set_weights` on layer \"{this.name}\"" +
                                            $"with a weight list of length {len(value)}, but the layer was " +
                                            $"expecting {len(weights)} weights.");
                foreach (var (this_w, v_w) in zip(weights, value))
                    this_w.assign(v_w, read_value: true);
            }
        }

        public List<IVariableV1> Variables => weights;

        public virtual LayerArgs get_config()
            => args;
    }
}
