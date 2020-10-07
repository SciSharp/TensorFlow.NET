using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class L2 : IRegularizer
    {
        float l2;

        public L2(float l2 = 0.01f)
        {
            this.l2 = l2;
        }

        public Tensor Apply(RegularizerArgs args)
        {
            throw new NotImplementedException();
        }
    }
}
