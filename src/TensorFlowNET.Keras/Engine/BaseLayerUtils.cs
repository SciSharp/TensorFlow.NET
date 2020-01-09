using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Initializers;
using Tensorflow.Keras.Metrics;

namespace Tensorflow.Keras.Engine
{
    public class BaseLayerUtils
    {
        public static (Metric, Metric) create_mean_metric(Tensor value, string name = null) => throw new NotImplementedException();

        public static VariableV1 make_variable(string name, TensorShape shape= null, TF_DataType dtype= TF_DataType.TF_FLOAT, Initializer initializer= null,
                            bool trainable= true, string caching_device= null, bool validate_shape= true, Constraints.ConstraintBase constraint= null,
                            bool use_resource= false, Graph[] collections= null, VariableSynchronization synchronization= VariableSynchronization.Auto,
                            VariableAggregation aggregation= VariableAggregation.None) => throw new NotImplementedException();

        public static Tensor[] collect_previous_mask(TensorArray input_tensors) => throw new NotImplementedException();

        public bool have_all_keras_metadata(Tensor[] tensors) => throw new NotImplementedException();

        public static dynamic generate_placeholders_from_shape(TensorShape shape) => throw new NotImplementedException();
    }
}
