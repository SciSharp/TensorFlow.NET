using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Metrics
{
    /// <summary>
    /// Encapsulates metric logic and state.
    /// </summary>
    public class Metric : Layer
    {
        public Metric(string name = null, TF_DataType dtype = TF_DataType.DtInvalid)
            : base(new LayerArgs
            {
                Name = name,
                DType = dtype
            })
        {
            stateful = true;
            built = true;
        }

        protected override IVariableV1 add_weight(string name, 
            TensorShape shape = null, 
            TF_DataType dtype = TF_DataType.TF_FLOAT, 
            IInitializer initializer = null, 
            IRegularizer regularizer = null, 
            VariableSynchronization synchronization = VariableSynchronization.OnRead, 
            VariableAggregation aggregation = VariableAggregation.Sum, 
            bool trainable = true, 
            Func<VariableArgs, IVariableV1> getter = null)
        {
            if (shape == null)
                shape = new TensorShape(new int[0]);

            return tf_with(ops.init_scope(), delegate
            {
                return base.add_weight(name, shape,
                    dtype: dtype,
                    trainable: false,
                    initializer: initializer,
                    synchronization: synchronization,
                    aggregation: aggregation);
            });
        }
    }
}
