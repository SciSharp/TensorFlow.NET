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

        public static Tensor accuracy(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor binary_accuracy(Tensor y_true, Tensor y_pred, float threshold = 0.5f) => throw new NotImplementedException();

        public static Tensor categorical_accuracy(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor sparse_categorical_accuracy(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor top_k_categorical_accuracy(Tensor y_true, Tensor y_pred, int k = 5) => throw new NotImplementedException();

        public static Tensor sparse_top_k_categorical_accuracy(Tensor y_true, Tensor y_pred, int k = 5) => throw new NotImplementedException();

        public static Tensor cosine_proximity(Tensor y_true, Tensor y_pred, int axis = -1) => throw new NotImplementedException();

        public static Metric clone_metric(Metric metric) => throw new NotImplementedException();

        public static Metric[] clone_metrics(Metric[] metric) => throw new NotImplementedException();

        public static string serialize(Metric metric) => throw new NotImplementedException();

        public static Metric deserialize(string config, object custom_objects = null) => throw new NotImplementedException();

        public static Metric get(object identifier) => throw new NotImplementedException();
    }
}
