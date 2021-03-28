using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Metrics
{
    /// <summary>
    /// Encapsulates metric logic and state.
    /// </summary>
    public class Metric : Layer
    {
        protected IVariableV1 total;
        protected IVariableV1 count;
        protected string _reduction;
        protected TF_DataType _dtype;

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

        public virtual Tensor update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
            => throw new NotImplementedException("");

        public virtual void reset_states()
        {
            foreach (var v in weights)
                v.assign(0);
        }

        public virtual Tensor result()
            => throw new NotImplementedException("");

        public override string ToString()
            => $"{name} {(float)total.numpy()}/{(float)count.numpy()}";
    }
}
