using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Regularizers
    {
        public IRegularizer l2(float l2 = 0.01f)
            => new L2(l2);
    }
}
