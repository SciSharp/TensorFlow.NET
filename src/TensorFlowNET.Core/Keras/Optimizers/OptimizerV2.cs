using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;
using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.Eager;

namespace Tensorflow.Keras.Optimizers
{
    /// <summary>
    /// Updated base class for optimizers.
    /// </summary>
    public class OptimizerV2 : Trackable, IOptimizer
    {
        protected bool _hypers_created;
        protected virtual string _name { get; }

        ResourceVariable _iterations;
        List<ResourceVariable> _weight;
        Dictionary<string, float> _hyper;
        Dictionary<string, ResourceVariable> _hyper_variables;
        protected bool _momentum;
        protected float _initial_decay = 0.0f;
        protected bool _use_locking = true;

        Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state;

        public OptimizerV2() : base()
        {
            _weight = new List<ResourceVariable>();
            _hyper = new Dictionary<string, float>();
            _hyper_variables = new Dictionary<string, ResourceVariable>();
            apply_state = new Dictionary<DeviceDType, Dictionary<string, Tensor>>();
        }

        public void apply_gradients((Tensor, ResourceVariable) grads_and_vars,
            string name = null,
            bool experimental_aggregate_gradients = true)
            => apply_gradients(grads_and_vars,
                name: name,
                experimental_aggregate_gradients: experimental_aggregate_gradients);

        /// <summary>
        /// Apply gradients to variables.
        /// </summary>
        /// <param name="grads_and_vars"></param>
        /// <param name="name"></param>
        /// <param name="experimental_aggregate_gradients"></param>
        public void apply_gradients(IEnumerable<(Tensor, ResourceVariable)> grads_and_vars,
            string name = null,
            bool experimental_aggregate_gradients = true)
        {
            var var_list = grads_and_vars.Select(x => x.Item2).ToArray();
            tf_with(ops.name_scope(_name), delegate
            {
                ops.init_scope();
                _create_all_weights(var_list);
                if (grads_and_vars == null || grads_and_vars.Count() == 0)
                    return control_flow_ops.no_op();

                apply_state = _prepare(var_list);
                if(experimental_aggregate_gradients)
                {
                    // var reduced_grads = _aggregate_gradients(grads_and_vars);
                    _distributed_apply(grads_and_vars, name, apply_state);
                }

                return null;
            });
        }

        void apply_grad_to_update_var(ResourceVariable var, EagerTensor grad)
        {
            _resource_apply_dense(var, grad, apply_state);
        }

        protected virtual Operation _resource_apply_dense(IVariableV1 var, 
            EagerTensor grad, 
            Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            throw new NotImplementedException("_resource_apply_dense");
        }

        void _distributed_apply(IEnumerable<(Tensor, ResourceVariable)> grads_and_vars,
            string name,
            Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            tf_with(ops.name_scope(name, "", new { skip_on_eager = true }), delegate
            {
                foreach(var (grad, var) in grads_and_vars)
                {
                    tf_with(ops.name_scope("update"), delegate
                    {
                        apply_grad_to_update_var(var, grad as EagerTensor);
                    });
                }

                _iterations.assign_add(ops.convert_to_tensor(1, dtype: _iterations.dtype));
            });
        }

        Tensor[] _aggregate_gradients(IEnumerable<(Tensor, ResourceVariable)> grads_and_vars)
        {
            return grads_and_vars.Select(x => x.Item1).ToArray();
        }

        Dictionary<DeviceDType, Dictionary<string, Tensor>> _prepare(IVariableV1[] var_list)
        {
            var _apply_state = new Dictionary<DeviceDType, Dictionary<string, Tensor>>();
            var keys = var_list.Select(x => new DeviceDType
            {
                Device = x.Device,
                DType = x.dtype.as_base_dtype()
            }).Distinct(new DeviceDType()).ToArray();

            foreach(var device_dtype in keys)
            {
                _apply_state[device_dtype] = new Dictionary<string, Tensor>();
                _prepare_local(device_dtype, _apply_state);
            }

            return _apply_state;
        }

        protected virtual void _prepare_local(DeviceDType device_dtype, 
            Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            if (_hyper.ContainsKey("learning_rate"))
            {
                var lr_t = array_ops.identity(_decayed_lr(device_dtype.DType));
                _apply_state[device_dtype]["lr_t"] = lr_t;
            }
        }

        Tensor _decayed_lr(TF_DataType var_dtype)
        {
            var lr_t = _get_hyper("learning_rate", var_dtype);
            if(_initial_decay > 0.0f)
            {
                throw new NotImplementedException("");
            }
            return lr_t;
        }

        protected ResourceVariable _get_hyper(string name, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            var value = _hyper_variables[name];
            return math_ops.cast(value, dtype);
        }

        void _create_all_weights(IVariableV1[] var_list)
        {
            if(_iterations == null)
            {
                _iterations = add_weight("iter", 
                    shape: new int[0], 
                    dtype: TF_DataType.TF_INT64, 
                    trainable: false, 
                    aggregation: VariableAggregation.OnlyFirstReplica);
                _weight.Add(_iterations);
            }

            _create_hypers();
            _create_slots(var_list);
        }

        protected void _set_hyper(string name, float value)
        {
            _hyper[name] = value;
        }

        void _create_hypers()
        {
            if (_hypers_created)
                return;
            foreach (var dict in _hyper)
            {
                var name = dict.Key;
                var value = dict.Value;
                _hyper_variables[name] = add_weight(
                    name,
                    shape: new int[0],
                    trainable: false,
                    initializer: tf.constant_initializer(value),
                    aggregation: VariableAggregation.OnlyFirstReplica);
            }
            _hypers_created = true;
        }

        void _create_slots(IVariableV1[] var_list)
        {
            if(_momentum)
            {
                /*for var in var_list:
                    self.add_slot(var, "momentum")*/
            }
        }

        ResourceVariable add_weight(string name,
            TensorShape shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            bool trainable = false,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            if (initializer == null)
                initializer = tf.zeros_initializer;

            if (dtype == TF_DataType.DtInvalid)
                dtype = TF_DataType.TF_FLOAT;

            var variable = _add_variable_with_custom_getter(new VariableArgs
            {
                Name = name,
                Shape = shape,
                Getter = base_layer_utils.make_variable,
                DType = dtype,
                Overwrite = true,
                Initializer = initializer,
                Trainable = trainable,
                UseResource = true,
                Synchronization = synchronization,
                Aggregation = aggregation
            });

            return variable as ResourceVariable;
        }
    }
}
