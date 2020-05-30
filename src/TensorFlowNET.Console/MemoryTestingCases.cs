using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class MemoryTestingCases
    {
        /// <summary>
        /// 
        /// </summary>
        public Action<int> Constant
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
                {
                    var tensor = tf.constant(3112.0f);
                }
            };
        public Action<int> Variable
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
                {
                    var tensor = tf.Variable(3112.0f);
                }
            };

        public Action<int> MathAdd
            => (iterate) =>
            {
                var x = tf.constant(3112.0f);
                var y = tf.constant(3112.0f);

                for (int i = 0; i < iterate; i++)
                {
                    var z = x + y;
                }
            };

        public Action<int> Gradient
            => (iterate) =>
            {
                for(int i = 0; i< iterate; i++)
                {
                    var w = tf.constant(3112.0f);
                    using var tape = tf.GradientTape();
                    tape.watch(w);
                    var loss = w * w;
                    var grad = tape.gradient(loss, w);
                }
            };
    }
}
