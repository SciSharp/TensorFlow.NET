using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Regularizers
{
    public class L1L2 : Regularizer
    {
        public L1L2(float l1 = 0f, float l2 = 0f)
        {
            throw new NotImplementedException();
        }

        public override float call(Tensor x)
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
