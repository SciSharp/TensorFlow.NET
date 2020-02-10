using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Nadam : Optimizer
    {
        public Nadam(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float schedule_decay = 0.004f) : base(null)
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
