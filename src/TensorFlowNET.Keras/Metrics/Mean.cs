using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Mean : Reduce
    {
        public Mean(string name, string dtype = null)
           : base(Reduction.MEAN, name, dtype)
        {
        }

    }
}
