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

using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Tensorflow.Eager;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using Tensorflow.Train;
using Tensorflow.Training;
using Tensorflow.Training.Saving.SavedModel;
using Tensorflow.Util;
using static Tensorflow.Binding;
using Tensorflow.Framework;
using Tensorflow.Sessions;
using Tensorflow.Common.Types;

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
        internal LayerArgs args;

        /// <summary>
        /// Indicates whether `build` needs to be called upon layer call, to create
        /// the layer's weights.
        /// </summary>
        protected bool built;
        public bool Built
        {
            get
            {
                return built;
            }
            internal set
            {
                built = value;
            }
        }
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
        public InputSpec InputSpec => inputSpec;
        bool dynamic = true;
        public bool SupportsMasking { get; set; }
        protected List<IVariableV1> _trainable_weights;

        public virtual List<IVariableV1> TrainableVariables => TrainableWeights;

        protected List<IVariableV1> _non_trainable_weights;
        public List<IVariableV1> NonTrainableVariables => NonTrainableWeights;
        public List<IVariableV1> Variables => Weights;

        public virtual List<IVariableV1> TrainableWeights
        {
            get
            {
                if (!this.Trainable)
                {
                    return new List<IVariableV1>();
                }
                var children_weights = _gather_children_variables(true);
                return children_weights.Concat(_trainable_weights).Distinct().ToList();
            }
        }

        public virtual List<IVariableV1> NonTrainableWeights
        {
            get
            {
                if (!this.Trainable)
                {
                    var children_weights = _gather_children_variables(true, true);
                    return children_weights.Concat(_trainable_weights).Concat(_non_trainable_weights).Distinct().ToList();
                }
                else
                {
                    var children_weights = _gather_children_variables(include_non_trainable: true);
                    return children_weights.Concat(_non_trainable_weights).Distinct().ToList();
                }
            }
        }

        public virtual List<IVariableV1> Weights
        {
            get
            {
                return TrainableWeights.Concat(NonTrainableWeights).ToList();
            }
            set
            {
                if (Weights.Count() != value.Count()) throw new ValueError(
                                            $"You called `set_weights` on layer \"{this.name}\"" +
                                            $"with a weight list of length {len(value)}, but the layer was " +
                                            $"expecting {len(Weights)} weights.");
                foreach (var (this_w, v_w) in zip(Weights, value))
                    this_w.assign(v_w, read_value: true);
            }
        }

        public virtual void set_weights(IEnumerable<NDArray> weights)
        {
            if (Weights.Count() != weights.Count()) throw new ValueError(
                                            $"You called `set_weights` on layer \"{this.name}\"" +
                                            $"with a weight list of length {len(weights)}, but the layer was " +
                                            $"expecting {len(Weights)} weights.");



            // check if the shapes are compatible
            var weight_index = 0;
            foreach(var w in weights)
            {
                if (!Weights[weight_index].AsTensor().is_compatible_with(w))
                {
                    throw new ValueError($"Layer weight shape {w.shape} not compatible with provided weight shape {Weights[weight_index].shape}");
                }
                weight_index++;
            }

            if (tf.executing_eagerly())
            {
                foreach (var (this_w, v_w) in zip(Weights, weights))
                    this_w.assign(v_w, read_value: true);
            }
            else
            {
                // TODO(Wanglongzhi2001):seems like there exist some bug in graph mode when define model, so uncomment the following when it fixed.

                //Tensors assign_ops = new Tensors();
                //var feed_dict = new FeedDict();

                //Graph g = tf.Graph().as_default();
                //foreach (var (this_w, v_w) in zip(Weights, weights))
                //{
                //    var tf_dtype = this_w.dtype;
                //    var placeholder_shape = v_w.shape;
                //    var assign_placeholder = tf.placeholder(tf_dtype, placeholder_shape);
                //    var assign_op = this_w.assign(assign_placeholder);
                //    assign_ops.Add(assign_op);
                //    feed_dict.Add(assign_placeholder, v_w);
                //}
                //var sess = tf.Session().as_default();
                //sess.run(assign_ops, feed_dict);

                //g.Exit();
            }
        }

        public List<NDArray> get_weights()
        {
            List<NDArray > weights = new List<NDArray>();
            weights.AddRange(Weights.ConvertAll(x => x.numpy())); 
            return weights;
        }

        protected int id;
        public int Id => id;
        protected string name;
        protected string base_name;
        public string Name
        {
            get
            {
                return name;
            }
            set
            {
                name = value;
            }
        }

        protected bool computePreviousMask;
        protected List<Operation> updates;
        public KerasShapesWrapper BatchInputShape => args.BatchInputShape;
        protected KerasShapesWrapper _buildInputShape = null;
        public KerasShapesWrapper BuildInputShape => _buildInputShape;

        List<INode> inboundNodes;
        public List<INode> InboundNodes => inboundNodes;
        List<INode> outboundNodes;
        public List<INode> OutboundNodes => outboundNodes;

        public Dictionary<string, object> SerializedAttributes { get; set; }

        ThreadLocal<CallContext> callContext = new ThreadLocal<CallContext>();
        public CallContext CallContext => callContext.Value;
        public Tensor[] input
        {
            get
            {
                if(inboundNodes is not null && inboundNodes.Count > 0)
                {
                    return inboundNodes[0].input_tensors;
                }
                return null;
            }
        }
        public Dictionary<int, List<INode>> NodesByDepth { get; set; }
        public Shape OutputShape
        {
            get
            {
                if(inboundNodes is not null && inboundNodes.Count > 0)
                {
                    return inboundNodes[0].Outputs.shape;
                }
                return null;
            }
        }
        protected List<ILayer> _self_tracked_trackables;

        /// <summary>
        /// If this value is set, the behavior of layer call will be changed to directly calling this function.
        /// </summary>
        public Func<Tensors, Tensors>? ReplacedCall { get; set; } = null;

        public Layer(LayerArgs args)
        {
            Initialize(args);
        }

        internal virtual void Initialize(LayerArgs args)
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
                args.BatchInputShape = new KerasShapesWrapper(new long[] { args.BatchSize }.Concat(args.InputShape.dims).ToArray());
            }
        }

        bool _in_functional_construction_mode(Tensors inputs)
        {
            return tf.Context.executing_eagerly()
                && inputs.Count(x => x is not EagerTensor && x is not NDArray) == inputs.Count() || _enforce_layer_construction;
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
        protected virtual Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            if(ReplacedCall is not null)
            {
                return ReplacedCall(inputs);
            }
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
               
            build(new KerasShapesWrapper(inputs.shape));

            if (need_restore_mode)
                tf.Context.restore_mode();

            built = true;
        }

        public virtual void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
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
                return layer_utils.count_params(this, Weights);
            return 0;
        }

        public virtual IKerasConfig get_config()
            => args;

        public virtual void adapt(Tensor data, int? batch_size = null, int? steps = null)
        {
            
        }

        public override void SetAttr(string name, object value)
        {
            // TODO(Rinne): deal with "_self_setattr_tracking".

            value = TrackableDataStructure.sticky_attribute_assignment(this, name, value);
            
            foreach(var val in nest.flatten(value))
            {
                if(val is Metric)
                {
                    // TODO(Rinne): deal with metrics.
                }
            }

            // TODO(Rinne): deal with "_auto_track_sub_layers".

            foreach(var val in nest.flatten(value))
            {
                if(val is not IVariableV1 variable)
                {
                    continue;
                }
                if (variable.Trainable)
                {
                    if (_trainable_weights.Contains(variable))
                    {
                        continue;
                    }
                    _trainable_weights.Add(variable);
                }
                else
                {
                    if (_non_trainable_weights.Contains(variable))
                    {
                        continue;
                    }
                    _non_trainable_weights.Add(variable);
                }
                keras.backend.track_variable(variable);
            }

            // Directly use the implementation of `Trackable`.
            var t = this.GetType();
            var field_info = t.GetField(name);
            if (field_info is not null)
            {
                field_info.SetValue(this, value);
            }
            else
            {
                CustomizedFields[name] = value;
            }
        }
    }
}
