using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Reduce : Metric
    {
        public Reduce(string reduction, string name, string dtype= null)
            : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override void update_state(Args args, KwArgs kwargs)
        {
            throw new NotImplementedException();
        }
    }
}
