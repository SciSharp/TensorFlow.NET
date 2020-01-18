using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Adamax : Optimizer
    {
        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float decay = 0) : base(null)
        {
            throw new NotImplementedException();
        }

        public override Tensor[] get_updates(Tensor loss, variables @params)
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
