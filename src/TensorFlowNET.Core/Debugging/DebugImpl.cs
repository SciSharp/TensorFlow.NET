using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Debugging
{
    public class DebugImpl
    {
        /// <summary>
        /// Set if device placements should be logged.
        /// </summary>
        /// <param name="enabled"> Whether to enabled device placement logging.</param>
        public void set_log_device_placement(bool enabled)
            => tf.Context.log_device_placement(enabled);

        /// <summary>
        /// Assert the condition `x == y` holds element-wise.
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="data"></param>
        /// <param name="message"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor assert_equal<T1, T2>(T1 t1,
            T2 t2,
            object[] data = null,
            string message = null,
            string name = null)
            => check_ops.assert_equal(t1,
                t2,
                data: data,
                message: message,
                name: name);

        public Tensor assert_greater_equal<T1, T2>(Tensor x,
            Tensor y,
            object[] data = null,
            string message = null,
            string name = null)
            => check_ops.assert_greater_equal(x,
                y,
                data: data,
                message: message,
                name: name);
    }
}
