using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Regularizers
{
    public abstract class Regularizer
    {
        public virtual float call(Tensor x)
        {
            return 0f;
        }

        public static Regularizer from_config(Hashtable hashtable) => throw new NotImplementedException();

        public virtual Hashtable get_config() => throw new NotImplementedException();

        public static Regularizer l1(float l = 0.01f)
        {
            return new L1L2(l1: l);
        }

        public static Regularizer l2(float l = 0.01f)
        {
            return new L1L2(l2: l);
        }

        public static Regularizer l1_l2(float l1 = 0.01f, float l2 = 0.01f)
        {
            return new L1L2(l1, l2);
        }

        public static string serialize(Regularizer regularizer) => throw new NotImplementedException();

        public static string deserialize(string config, dynamic custom_objects = null) => throw new NotImplementedException();

        public static Regularizer get(object identifier) => throw new NotImplementedException();
    }
}
