namespace Tensorflow.Keras.Engine;

public interface IOptimizer
{
    Tensor[] aggregate_gradients(IEnumerable<(Tensor, IVariableV1)> grads_and_vars);
    Tensor[] clip_gradients(Tensor[] grads);
    void apply_gradients((Tensor, IVariableV1) grads_and_vars,
            string name = null,
            bool experimental_aggregate_gradients = true);
    void apply_gradients(IEnumerable<(Tensor, IVariableV1)> grads_and_vars,
            string name = null,
            bool experimental_aggregate_gradients = true);
}
