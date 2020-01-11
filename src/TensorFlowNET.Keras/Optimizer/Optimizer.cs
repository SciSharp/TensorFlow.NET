using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Optimizer
    {
        public Optimizer(KwArgs kwargs)
        {
            throw new NotImplementedException();
        }

        public virtual Tensor[] get_updates(Tensor loss, variables @params)
        {
            return null;
        }

        public virtual Tensor[] get_gradients(Tensor loss, variables @params) => throw new NotImplementedException();

        public virtual void set_weights(NDArray[] weights) => throw new NotImplementedException();

        public virtual NDArray[] get_weights() => throw new NotImplementedException();

        public virtual Hashtable get_config() => throw new NotImplementedException();

        public static string serialize(Optimizer optimizer) => throw new NotImplementedException();

        public static string deserialize(string config, object custom_objects = null) => throw new NotImplementedException();

        public static Optimizer get(object identifier) => throw new NotImplementedException();

    }
}
