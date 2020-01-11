using Keras.Layers;
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

        public Layer[] create_keras_history(Tensor[] tensors) => throw new NotImplementedException();   

        private void _create_keras_history_helper(Tensor[] tensors, TensorFlowOpLayer[] processed_ops, Layer[] created_layers) => throw new NotImplementedException();

        public Tensor[] unnest_if_single_tensor(Tensor[] input_tensors) => throw new NotImplementedException();

        public bool needs_keras_history(Tensor[] tensors, bool ignore_call_context= false) => throw new NotImplementedException();

        public bool is_in_keras_graph() => throw new NotImplementedException();

        public string is_in_eager_or_tf_function() => throw new NotImplementedException();

        public bool is_in_tf_function() => throw new NotImplementedException();

        public bool uses_keras_history(Tensor[] tensors) => throw new NotImplementedException();

        public Tensor[] mark_checked(Tensor[] tensors) => throw new NotImplementedException();

        public CallContext call_context() => throw new NotImplementedException();
    }
}
