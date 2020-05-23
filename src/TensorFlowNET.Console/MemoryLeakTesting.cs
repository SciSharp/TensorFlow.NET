using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class MemoryLeakTesting
    {
        public void WarmUp()
        {
            print(tf.VERSION);
        }

        /// <summary>
        /// 
        /// </summary>
        public void TensorCreation()
        {
            int total = 10 * 1000 * 1000;
            for(int i = 0; i < total; i++)
            {
                var const1 = tf.constant(3112.0f);
                // const1.Dispose();
            }
        }
    }
}
