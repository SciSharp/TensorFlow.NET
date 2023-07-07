namespace Tensorflow.Keras.Optimizers
{
    public class AdamW : Adam
    {
        string name;
        float weight_decay;
        DeviceDType deType;
        List<string> no_decay_params = null;
        public AdamW(float learning_rate= 0.001f,
                     float weight_decay= 0.004f,
                     float beta_1= 0.9f,
                     float beta_2= 0.999f,
                     float epsilon= 1e-7f,
                     bool amsgrad = false,
                     List<string> no_decay_params = null,
                     string name= "AdamW") : base(learning_rate, beta_1, beta_2, epsilon, amsgrad)
        {
            this.name = name;
            this.weight_decay = weight_decay;
            this.no_decay_params = no_decay_params;
        }

        protected Operation _decay_weights_op(IVariableV1 var, float learning_rate, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            bool do_decay = _do_use_weight_decay(var.Name);
            if (do_decay) return var.assign_add(
                -learning_rate * var.AsTensor() * apply_state[deType]["weight_decay"]);
            return tf.no_op();
        }


        protected bool _do_use_weight_decay(string param_name)
        {
            // Whether to use L2 weight decay for `param_name`.
            if (this.weight_decay == 0)
                return false;

            if (this.no_decay_params != null)
            {
                foreach (var name in no_decay_params)
                {
                    if (param_name.Contains(name)) return false;
                }

            }
            return true;
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            var decay = _decay_weights_op(var, _hyper["learning_rate"], apply_state);
            tf.control_dependencies(new[] { decay });
            return base._resource_apply_dense(var, grad, apply_state);
        }

        protected override void _prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            this.deType = device_dtype;
            base._prepare_local(device_dtype, apply_state);
            apply_state[device_dtype]["weight_decay"] = tf.constant(
                weight_decay, name: "adam_weight_decay_rate");
        }
    }
}
