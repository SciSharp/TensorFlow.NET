using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Adam : Optimizer
    {
        public Adam(float lr= 0.001f, float beta_1 = 0.9f, float beta_2 = 0.99f, float? epsilon = null, float decay = 0) : base(null)
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
