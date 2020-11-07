using System;

namespace Tensorflow.Training
{
    public class learning_rate_decay
    {
        /// <summary>
        /// Applies a polynomial decay to the learning rate.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="global_step"></param>
        /// <param name="decay_steps"></param>
        /// <param name="end_learning_rate"></param>
        /// <param name="power"></param>
        /// <param name="cycle"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor polynomial_decay(float learning_rate, RefVariable global_step, float decay_steps,
            float end_learning_rate = 0.0001f, float power = 1.0f, bool cycle = false,
            string name = null)
        {
            throw new NotImplementedException("");
        }
    }
}
