using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Base layer class.
    /// A layer is a class implementing common neural networks operations, such
    /// as convolution, batch norm, etc. These operations require managing weights,
    /// losses, updates, and inter-layer connectivity.
    /// </summary>
    public class Layer : CheckpointableBase
    {
        protected bool trainable;
        protected string _name;
        protected TF_DataType _dtype;
        protected Graph _graph;
        protected string _base_name;
        protected VariableScope _scope;
        /// <summary>
        /// A stateful layer is a layer whose updates are run during inference too,
        /// for instance stateful RNNs.
        /// </summary>
        protected bool stateful;
        /// <summary>
        /// Indicates whether `build` needs to be called upon layer call, to create
        /// the layer's weights.
        /// </summary>
        protected bool built;
        /// <summary>
        /// Provides information about which inputs are compatible with the layer.
        /// </summary>
        protected InputSpec input_spec;
        protected bool supports_masking;

        public Layer(bool trainable = true, 
            string name = null, 
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            this.trainable = trainable;
            this.stateful = false;
            this.built = false;
            this.supports_masking = false;
            _init_set_name(name);
        }

        public Tensor apply(Tensor inputs)
        {
            return __call__(inputs);
        }

        public Tensor __call__(Tensor inputs,
            VariableScope scope = null)
        {
            _set_scope(scope);
            _graph = ops._get_graph_from_inputs(new List<Tensor> { inputs }, graph: _graph);
            var scope_context_manager = tf.variable_scope(_scope);

            throw new NotImplementedException("");
        }

        private void _init_set_name(string name)
        {
            if (string.IsNullOrEmpty(name))
                (_name, _base_name) = _make_unique_name();
        }

        private (string, string) _make_unique_name()
        {
            string base_name = "conv2d";
            string name = base_layer_utils.unique_layer_name(base_name);
            return (name, base_name);
        }

        private void _set_scope(VariableScope scope = null)
        {
            if (_scope == null)
            {
                Python.with(tf.variable_scope(scope, default_name: _base_name), captured_scope =>
                {
                    _scope = captured_scope;
                });
            }
                
        }
    }
}
