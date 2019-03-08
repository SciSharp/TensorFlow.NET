using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Layers
{
    public class Layer : Keras.Engine.Layer
    {
        protected bool trainable;
        protected string _name;
        protected TF_DataType _dtype;
        protected Graph _graph;
        protected string _base_name;
        protected VariableScope _scope;
        protected VariableScope _current_scope;
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
        protected bool? _reuse;

        public Layer(bool trainable = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool? _reuse = null)
        {
            this.trainable = trainable;
            this.stateful = false;
            this._reuse = _reuse;
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

            variable_scope scope_context_manager = null;
            if (built)
            {

            }
            else
            {
                scope_context_manager = tf.variable_scope(_scope,
                    auxiliary_name_scope: false);
            }

            Python.with(scope_context_manager, scope2 => _current_scope = scope2);
            // Actually call layer
            var outputs = base.__call__(inputs);

            throw new NotImplementedException("");
        }

        protected virtual void add_weight(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool? trainable = null)
        {
            var default_graph = ops.get_default_graph();
            Graph init_graph = null;
            RefVariable[] existing_variables = null;

            if (default_graph.building_function)
            {
                throw new NotImplementedException("add_weight");
            }
            else
            {
                init_graph = default_graph;
                existing_variables = variables.global_variables().ToArray();
            }

            if(dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            _set_scope();
            var reuse = built || (_reuse != null && _reuse.Value);
            Python.with(tf.variable_scope(_scope, 
                reuse: reuse, 
                auxiliary_name_scope: false), scope =>
            {
                _current_scope = scope;
                Python.with(ops.name_scope(_name_scope()), delegate
                {
                    base.add_weight(name,
                        shape,
                        dtype: dtype,
                        initializer: initializer,
                        trainable: trainable,
                        getter: (name1, shape1, dtype1, initializer1, trainable1) =>
                        {
                            return tf.get_variable(name1, 
                                shape: new TensorShape(shape1),
                                dtype: dtype1,
                                initializer: initializer1,
                                trainable: trainable1);
                        });
                });
            });
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

        protected override string _name_scope()
        {
            return _current_scope.original_name_scope;
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
