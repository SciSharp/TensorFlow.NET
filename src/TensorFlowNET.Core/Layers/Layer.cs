using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Layers
{
    public class Layer : Keras.Engine.Layer
    {
        protected Graph _graph;
        
        protected VariableScope _scope;
        protected VariableScope _current_scope;
        
        protected bool? _reuse;
        protected bool _use_resource_variables;
        protected bool _keras_style;

        public Layer(bool trainable = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool? _reuse = null) : base(trainable: trainable, name: name, dtype: dtype)
        {
            this._use_resource_variables = false;
            this._reuse = _reuse;
            this.built = false;
            _keras_style = false;
        }

        public virtual Tensor apply(Tensor inputs, Tensor training = null)
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

            // Update global default collections.
            //_add_elements_to_collection(updates, ops.GraphKeys.UPDATE_OPS);

            return outputs;
        }

        protected virtual RefVariable add_weight(string name,
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
            return Python.with(tf.variable_scope(_scope, 
                reuse: reuse, 
                auxiliary_name_scope: false), scope =>
            {
                _current_scope = scope;
                return Python.with(ops.name_scope(_name_scope()), delegate
                {
                    var variable = base.add_weight(name,
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

                    if(init_graph != null)
                    {
                        var trainable_variables = variables.trainable_variables();
                    }
                    return variable;
                });
            });
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
