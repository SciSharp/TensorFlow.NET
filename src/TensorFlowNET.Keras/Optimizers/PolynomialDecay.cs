using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Optimizers
{
    /// <summary>
    /// A LearningRateSchedule that uses a polynomial decay schedule.
    /// </summary>
    public class PolynomialDecay : LearningRateSchedule
    {
        float initial_learning_rate;
        float decay_steps;
        float end_learning_rate;
        float power;
        bool cycle;
        string name;

        public PolynomialDecay(float initial_learning_rate,
            float decay_steps,
            float end_learning_rate = 0.0001f,
            float power = 1.0f,
            bool cycle = false,
            string name = null) : base()
        {
            this.initial_learning_rate = initial_learning_rate;
            this.decay_steps = decay_steps;
            this.end_learning_rate = end_learning_rate;
            this.power = power;
            this.cycle = cycle;
            this.name = name;
        }

        public Tensor __call__(IVariableV1 step)
        {
            return tf_with(ops.name_scope(name ?? "PolynomialDecay"), scope =>
            {
                name = scope;
                var initial_learning_rate_tensor = ops.convert_to_tensor(initial_learning_rate, name: "initial_learning_rate");
                var dtype = initial_learning_rate_tensor.dtype;
                var end_learning_rate_tensor = constant_op.constant(end_learning_rate, dtype);
                var power_tensor = constant_op.constant(power, dtype);

                var global_step_recomp = constant_op.constant(step, dtype);
                var decay_steps_recomp = constant_op.constant(decay_steps, dtype);

                if (cycle)
                {
                    throw new NotImplementedException("PolynomialDecay cycle");
                }
                else
                {
                    // Make sure that the global_step used is not bigger than decay_steps.
                    global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps);
                }

                var p = tf.divide(global_step_recomp, decay_steps_recomp);
                var pow = tf.pow(1 - p, power_tensor);
                var m = math_ops.multiply(initial_learning_rate_tensor - end_learning_rate_tensor, pow);
                return math_ops.add(m,
                  end_learning_rate_tensor,
                  name: name);
            });
        }
    }
}
