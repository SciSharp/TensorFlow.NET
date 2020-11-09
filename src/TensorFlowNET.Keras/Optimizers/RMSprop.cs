using System;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Optimizers
{
    /// <summary>
    /// Optimizer that implements the RMSprop algorithm.
    /// </summary>
    public class RMSprop : OptimizerV2
    {
        RMSpropArgs args;
        bool centered => args.Centered;
        protected override string _name => "RMSprop";

        public RMSprop(RMSpropArgs args) : base(args)
        {
            this.args = args;
            _set_hyper("rho", args.RHO);
            _set_hyper("momentum", args.Momentum);
        }

        protected override void _create_slots(IVariableV1[] var_list)
        {
            foreach (var var in var_list)
                add_slot(var, "rms");
            if (_momentum)
                foreach (var var in var_list)
                    add_slot(var, "momentum");
            if (centered)
                foreach (var var in var_list)
                    add_slot(var, "mg");
        }

        protected override void _prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            base._prepare_local(device_dtype, _apply_state);
            var rho = array_ops.identity(_get_hyper("rho", device_dtype.DType));
            _apply_state[device_dtype]["neg_lr_t"] = -_apply_state[device_dtype]["lr_t"];
            _apply_state[device_dtype]["epsilon"] = ops.convert_to_tensor(args.Epsilon, dtype: device_dtype.DType);
            _apply_state[device_dtype]["rho"] = rho;
            _apply_state[device_dtype]["momentum"] = array_ops.identity(_get_hyper("momentum", device_dtype.DType));
            _apply_state[device_dtype]["one_minus_rho"] = 1.0f - rho;
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            Dictionary<string, Tensor> coefficients = null;
            foreach (var state in _apply_state)
            {
                if (state.Key.DType == var.dtype.as_base_dtype()
                    && state.Key.Device == var.Device)
                {
                    coefficients = state.Value;
                    break;
                }
            }

            var rms = get_slot(var, "rms");
            if (_momentum)
            {
                throw new NotImplementedException("");
            }
            else
            {
                var rms_t = coefficients["rho"] * rms.AsTensor() + coefficients["one_minus_rho"] * math_ops.square(grad);
                rms_t = state_ops.assign(rms, rms_t, use_locking: _use_locking);
                var denom_t = rms_t;
                if (centered)
                {
                    throw new NotImplementedException("");
                }
                var var_t = var.AsTensor() - coefficients["lr_t"] * grad / (math_ops.sqrt(denom_t) + coefficients["epsilon"]);
                return state_ops.assign(var, var_t, use_locking: _use_locking).op;
            }
        }
    }
}
