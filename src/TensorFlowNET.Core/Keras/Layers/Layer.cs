using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Base layer class.
    /// A layer is a class implementing common neural networks operations, such
    /// as convolution, batch norm, etc. These operations require managing weights,
    /// losses, updates, and inter-layer connectivity.
    /// </summary>
    public class Layer : CheckpointableBase
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
        protected List<RefVariable> _trainable_weights;
        public string _name;
        protected string _base_name;
        protected bool _compute_previous_mask;
        protected List<Operation> _updates;
        public int[] _batch_input_shape;

        private List<Node> _inbound_nodes;
        public List<Node> inbound_nodes => _inbound_nodes;

        private List<Node> _outbound_nodes;
        public List<Node> outbound_nodes => _outbound_nodes;

        public Layer(bool trainable = true, 
            string name = null, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] input_shape = null)
        {
            this.trainable = trainable;
            this._dtype = dtype;
            stateful = false;
            built = false;
            this.supports_masking = false;
            _init_set_name(name);
            _trainable_weights = new List<RefVariable>();
            _compute_previous_mask = false;
            _updates = new List<Operation>();

            // Manage input shape information if passed.

            _batch_input_shape = new int[] { -1, -1 };

            _dtype = dtype;

            _inbound_nodes = new List<Node>();
        }

        public Tensor __call__(Tensor[] inputs,
            Tensor training = null,
            VariableScope scope = null)
        {
            var input_list = inputs;
            Tensor outputs = null;

            // We will attempt to build a TF graph if & only if all inputs are symbolic.
            // This is always the case in graph mode. It can also be the case in eager
            // mode when all inputs can be traced back to `keras.Input()` (when building
            // models using the functional API).
            bool build_graph = tf_utils.are_all_symbolic_tensors(input_list);

            // Handle Keras mask propagation from previous layer to current layer.
            Python.with(ops.name_scope(_name_scope()), delegate
            {
                if (!built)
                {
                    _maybe_build(inputs);
                    built = true;
                }

                if (build_graph)
                {
                    // Symbolic execution on symbolic tensors. We will attempt to build
                    // the corresponding TF subgraph inside `backend.get_graph()`
                    var graph = backend.get_graph();
                    outputs = call(inputs[0], training: training);
                    _handle_activity_regularization(inputs[0], outputs);
                    _set_mask_metadata(inputs[0], outputs, null);
                }
            });

            return outputs;
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
            return null;
        }

        protected void _maybe_build(Tensor[] inputs)
        {
            var input_list = inputs;
            build(input_list[0].GetShape());
        }

        protected virtual void build(TensorShape input_shape)
        {
            built = true;
        }

        protected virtual RefVariable add_weight(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null,
            Func<string, int[], TF_DataType, IInitializer, bool, RefVariable> getter = null)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            if (trainable == null)
                trainable = true;

            var variable = _add_variable_with_custom_getter(name,
                shape,
                dtype: dtype,
                getter: getter == null ? base_layer_utils.make_variable : getter,
                overwrite: true,
                initializer: initializer,
                trainable: trainable.Value);
            backend.track_variable(variable);
            _trainable_weights.Add(variable);

            return variable;
        }

        protected virtual void add_update(Tensor[] updates, bool inputs = false)
        {
            var updates_op = updates.Select(x => x.op).ToArray();
            _updates.AddRange(updates_op);
        }

        protected virtual void _init_set_name(string name)
        {
            string base_name = name;
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
