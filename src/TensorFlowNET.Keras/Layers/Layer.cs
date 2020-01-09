using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras.Constraints;
using Tensorflow.Keras.Initializers;
using Tensorflow.Keras.Regularizers;

namespace Keras.Layers
{
    public abstract class Layer
    {
        public Layer(bool trainable = true, string name = null, string dtype = null, bool @dynamic = false, Dictionary<string, object> kwargs = null)
        {

        }

        public void build(TensorShape shape) => throw new NotImplementedException();

        public void call(Tensor[] inputs) => throw new NotImplementedException();

        public void _add_trackable(dynamic trackable_object, bool trainable) => throw new NotImplementedException();

        public void add_weight(string name= null, TensorShape shape= null, string dtype= null, Initializer initializer = null,
                               Regularizer regularizer = null, bool? trainable = null, ConstraintBase constraint = null,
                                dynamic partitioner= null, bool? use_resource= null, VariableSynchronization synchronization= VariableSynchronization.Auto,
                                VariableAggregation aggregation= VariableAggregation.None, Dictionary<string, object> kwargs = null) => throw new NotImplementedException();

        public Dictionary<string, object> get_config() => throw new NotImplementedException();

        public Layer from_config(Dictionary<string, object> config) => throw new NotImplementedException();

        public TensorShape compute_output_shape(TensorShape input_shape) => throw new NotImplementedException();

        public dynamic compute_output_signature(dynamic input_signature) => throw new NotImplementedException();
    }
}
