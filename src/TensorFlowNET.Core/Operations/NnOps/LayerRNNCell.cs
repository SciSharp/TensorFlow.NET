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
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow
{
    [Obsolete("This is an incompleted tf v1 api, pleas use keras RNNs instead.")]
    public class LayerRnnCell : RnnCell
    {
        protected InputSpec inputSpec;
        protected bool built;
        protected Graph _graph;

        protected VariableScope _scope;
        protected VariableScope _current_scope;

        protected bool? _reuse;
        protected bool _use_resource_variables;
        protected bool _keras_style;

        public LayerRnnCell(bool trainable = true,
                    string name = null,
                    TF_DataType dtype = TF_DataType.DtInvalid,
                    bool? _reuse = null) : base(_reuse: _reuse,
                    name: name,
                    dtype: dtype)
        {
            // For backwards compatibility, legacy layers do not use `ResourceVariable`
            // by default.
            this._use_resource_variables = false;
            this._reuse = _reuse;

            // Avoid an incorrect lint error
            this.built = false;
            _keras_style = false;
        }

        protected virtual void build(Shape inputs_shape)
        {

        }

        public virtual (Tensor, Tensor) apply(Tensor inputs, Tensor training = null)
        {
            var results = __call__(inputs, training: training);
            return (results[0], results[1]);
        }

        public Tensors __call__(Tensors inputs,
            Tensor state = null,
            Tensor training = null,
            VariableScope scope = null)
        {
            _set_scope(scope);
            _graph = ops._get_graph_from_inputs(inputs, graph: _graph);

            variable_scope scope_context_manager = null;
            if (built)
            {
                scope_context_manager = tf.variable_scope(_scope,
                    reuse: true,
                    auxiliary_name_scope: false);
            }
            else
            {
                scope_context_manager = tf.variable_scope(_scope,
                    reuse: _reuse,
                    auxiliary_name_scope: false);
            }

            Tensors outputs = null;
            tf_with(scope_context_manager, scope2 =>
            {
                _current_scope = scope2;
                // Actually call layer

            });


            // Update global default collections.

            return outputs;
        }

        protected virtual void _add_elements_to_collection(Operation[] elements, string[] collection_list)
        {
            foreach (var name in collection_list)
            {
                var collection = ops.get_collection_ref<Operation>(name);

                foreach (var element in elements)
                    if (!collection.Contains(element))
                        collection.Add(element);
            }
        }

        /// <summary>
        /// Adds a new variable to the layer, or gets an existing one; returns it.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="initializer"></param>
        /// <param name="trainable"></param>
        /// <param name="synchronization"></param>
        /// <param name="aggregation"></param>
        /// <returns></returns>
        protected virtual IVariableV1 add_weight(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool trainable = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            var default_graph = ops.get_default_graph();
            Graph init_graph = null;
            IVariableV1[] existing_variables = null;

            if (synchronization == VariableSynchronization.OnRead)
                trainable = false;

            if (default_graph.building_function)
            {
                throw new NotImplementedException("add_weight");
            }
            else
            {
                init_graph = default_graph;
                existing_variables = variables.global_variables().ToArray();
            }

            if (dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            _set_scope();
            var reuse = built || (_reuse != null && _reuse.Value);
            return tf.Variable(0);
        }

        protected string _name_scope()
        {
            return _current_scope.original_name_scope;
        }

        protected void _set_scope(VariableScope scope = null)
        {
            if (_scope == null)
            {
                if (_reuse.HasValue && _reuse.Value)
                {
                    throw new NotImplementedException("_set_scope _reuse.HasValue");
                    /*with(tf.variable_scope(scope == null ? _base_name : scope),
                        captured_scope => _scope = captured_scope);*/
                }
                else
                {

                }
            }
        }
    }
}
