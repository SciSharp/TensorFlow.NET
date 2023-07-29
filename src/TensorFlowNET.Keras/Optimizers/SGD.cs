using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Optimizers
{
    public class SGD : OptimizerV2
    {
        protected override string _name => "SGD";

#pragma warning disable CS0169 // The field 'SGD.nesterov' is never used
        bool nesterov;
#pragma warning restore CS0169 // The field 'SGD.nesterov' is never used

        public SGD(float learning_rate,
            float momentum = 0.0f,
            bool nesterov = false,
            float decay = 0.0f) : base(new OptimizerV2Args { })
        {
            _set_hyper("learning_rate", learning_rate);
            _set_hyper("decay", decay);

            _momentum = momentum > 0;
            if (momentum < 0 || momentum > 1)
                throw new ValueError($"momentum must be a number between 0 and 1, got {momentum}.");

            _set_hyper("momentum", momentum);

#pragma warning disable CS1717 // Assignment made to same variable
            nesterov = nesterov;
#pragma warning restore CS1717 // Assignment made to same variable
        }

        protected override void _create_slots(IVariableV1[] var_list)
        {
            if (_momentum)
                foreach (var var in var_list)
                    add_slot(var, "momentum");
        }

        protected override void _prepare_local(DeviceDType device_dtype,
            Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            base._prepare_local(device_dtype, _apply_state);

            _apply_state[device_dtype]["momentum"] = array_ops.identity(
                _get_hyper("momentum", device_dtype.DType));
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            if (_momentum)
            {
                var momentum_var = get_slot(var, "momentum");
                return gen_training_ops.resource_apply_keras_momentum(
                    var.Handle,
                    momentum_var.Handle,
                    _get_hyper("learning_rate", var.dtype),
                    grad,
                    _get_hyper("momentum", var.dtype),
                    use_locking: _use_locking,
                    use_nesterov: nesterov);
            }
            var device_dtype = _apply_state.Keys.FirstOrDefault(x => x.Device == var.Device && x.DType == var.dtype.as_base_dtype());

            return gen_training_ops.resource_apply_gradient_descent(var.Handle,
                _apply_state[device_dtype]["lr_t"],
                grad,
                use_locking: _use_locking);
        }
    }
}
