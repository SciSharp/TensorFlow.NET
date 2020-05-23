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
            int total = 1 * 1000 * 1000;
            for (int i = 0; i < total; i++)
            {
                /*var const1 = new Tensor(new float[,]
                 {
                    { 3.0f, 1.0f },
                    { 1.0f, 2.0f }
                 });
                const1.Dispose();*/

                var tensor = new EagerTensorV2(new float[,]
                 {
                    { 3.0f, 1.0f },
                    { 1.0f, 2.0f }
                 });

                tensor.Dispose();
            }

            GC.Collect();
        }
    }
}
