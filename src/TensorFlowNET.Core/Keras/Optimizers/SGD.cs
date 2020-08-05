using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;

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
            float decay = 0.0f) : base()
        {
            _set_hyper("learning_rate", learning_rate);
            _set_hyper("decay", decay);

            _momentum = momentum > 0;

            _set_hyper("momentum", momentum);

#pragma warning disable CS1717 // Assignment made to same variable
            nesterov = nesterov;
#pragma warning restore CS1717 // Assignment made to same variable
        }

        protected override void _prepare_local(DeviceDType device_dtype, 
            Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            base._prepare_local(device_dtype, _apply_state);

            _apply_state[device_dtype]["momentum"] = array_ops.identity(
                _get_hyper("momentum", device_dtype.DType));
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, EagerTensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            if (_momentum)
            {
                throw new NotImplementedException("_resource_apply_dense");
            }
            var device_dtype = _apply_state.Keys.FirstOrDefault(x => x.Device == var.Device && x.DType == var.dtype.as_base_dtype());

            return gen_training_ops.resource_apply_gradient_descent(var.Handle, 
                _apply_state[device_dtype]["lr_t"], 
                grad,
                use_locking: _use_locking);
        }
    }
}
