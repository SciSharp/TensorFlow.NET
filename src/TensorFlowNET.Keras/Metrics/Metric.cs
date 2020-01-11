using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public abstract class Metric : Layers.Layer
    {
        public string dtype
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public Metric(string name, string dtype)
        {
            throw new NotImplementedException();
        }

        public void __new__ (Metric cls, Args args, KwArgs kwargs) => throw new NotImplementedException();

        public Tensor __call__(Metric cls, Args args, KwArgs kwargs) => throw new NotImplementedException();

        public virtual Hashtable get_config() => throw new NotImplementedException();

        public virtual void reset_states() => throw new NotImplementedException();

        public abstract void update_state(Args args, KwArgs kwargs);

        public abstract Tensor result();

        public void add_weight(string name, TensorShape shape= null, VariableAggregation aggregation= VariableAggregation.Sum,
                                VariableSynchronization synchronization = VariableSynchronization.OnRead, Initializers.Initializer initializer= null, 
                                string dtype= null) => throw new NotImplementedException();
    }
}
