using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Optimizers
{
    /// <summary>
    /// Optimizer that implements the Adam algorithm.
    /// Adam optimization is a stochastic gradient descent method that is based on
    /// adaptive estimation of first-order and second-order moments.
    /// </summary>
    public class Adam : OptimizerV2
    {
        protected override string _name => "Adam";
        float epsilon = 1e-7f;
        bool amsgrad = false;

        public Adam(float learning_rate = 0.001f,
                float beta_1 = 0.9f,
                float beta_2 = 0.999f,
                float epsilon = 1e-7f,
                bool amsgrad = false,
                string name = "Adam") : base(new OptimizerV2Args { })
        {
            _set_hyper("learning_rate", learning_rate);
            // _set_hyper("decay", _initial_decay);
            _set_hyper("beta_1", beta_1);
            _set_hyper("beta_2", beta_2);
            this.epsilon = epsilon;
            this.amsgrad = amsgrad;
        }

        protected override void _create_slots(IVariableV1[] var_list)
        {
            foreach (var var in var_list)
                add_slot(var, "m");
            foreach (var var in var_list)
                add_slot(var, "v");
            if (amsgrad)
                foreach (var var in var_list)
                    add_slot(var, "vhat");
        }

        protected override void _prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            base._prepare_local(device_dtype, apply_state);
            var var_dtype = device_dtype.DType;
            var var_device = device_dtype.Device;
            var local_step = math_ops.cast(iterations + 1, var_dtype);
            var beta_1_t = array_ops.identity(_get_hyper("beta_1", var_dtype));
            var beta_2_t = array_ops.identity(_get_hyper("beta_2", var_dtype));
            var beta_1_power = math_ops.pow(beta_1_t, local_step);
            var beta_2_power = math_ops.pow(beta_2_t, local_step);
            var lr = apply_state[device_dtype]["lr_t"] * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power));
            // update state
            apply_state[device_dtype]["lr"] = lr;
            apply_state[device_dtype]["epsilon"] = ops.convert_to_tensor(epsilon);
            apply_state[device_dtype]["beta_1_t"] = beta_1_t;
            apply_state[device_dtype]["beta_1_power"] = beta_1_power;
            apply_state[device_dtype]["one_minus_beta_1_t"] = 1 - beta_1_t;
            apply_state[device_dtype]["beta_2_t"] = beta_2_t;
            apply_state[device_dtype]["beta_2_power"] = beta_2_power;
            apply_state[device_dtype]["one_minus_beta_2_t"] = 1 - beta_2_t;
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            var (var_device, var_dtype) = (var.Device, var.dtype.as_base_dtype());
            var coefficients = apply_state.FirstOrDefault(x => x.Key.Device == var_device && x.Key.DType == var_dtype).Value ?? _fallback_apply_state(var_device, var_dtype);
            var m = get_slot(var, "m");
            var v = get_slot(var, "v");

            if (!amsgrad)
                return gen_training_ops.resource_apply_adam(var.Handle,
                    m.Handle,
                    v.Handle,
                    coefficients["beta_1_power"],
                    coefficients["beta_2_power"],
                    coefficients["lr_t"],
                    coefficients["beta_1_t"],
                    coefficients["beta_2_t"],
                    coefficients["epsilon"],
                    grad,
                    use_locking: _use_locking);
            else
                throw new NotImplementedException("");
        }
    }
}
