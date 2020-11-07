using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public class moving_averages
    {
        /// <summary>
        /// Compute the moving average of a variable.
        /// </summary>
        /// <param name="variable"></param>
        /// <param name="value"></param>
        /// <param name="decay"></param>
        /// <param name="zero_debias"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor assign_moving_average(IVariableV1 variable, IVariableV1 value, Tensor decay,
            bool zero_debias = true, string name = null)
        {
            return tf_with(ops.name_scope(name, "AssignMovingAvg", new { variable, value, decay }), scope =>
            {
                decay = ops.convert_to_tensor(1.0f - decay, name: "decay");
                if (decay.dtype != variable.dtype.as_base_dtype())
                    decay = math_ops.cast(decay, variable.dtype.as_base_dtype());

                return state_ops.assign_sub(variable, (variable.AsTensor() - value.AsTensor()) * decay, name: scope);
            });
        }
    }
}
