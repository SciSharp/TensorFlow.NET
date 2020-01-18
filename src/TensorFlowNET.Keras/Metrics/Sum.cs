using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Sum : Reduce
    {
        public Sum(string name, string dtype = null)
           : base(Reduction.SUM, name, dtype)
        {
        }
    }
}
