using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanMetricWrapper : Mean
    {
        public MeanMetricWrapper(Func<Tensor, Tensor, Tensor> fn, string name, string dtype = null) : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
