using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Losses
{
    public class LossFunctionWrapper : Loss
    {
        Action fn;

        public LossFunctionWrapper(Action fn,
            string reduction = ReductionV2.AUTO, 
            string name = null) 
            : base(reduction: reduction, 
                  name: name)
        {
            this.fn = fn;
        }
    }
}
