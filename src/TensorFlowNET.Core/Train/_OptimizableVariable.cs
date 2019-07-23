namespace Tensorflow
{
    public interface _OptimizableVariable
    {
        Tensor target();
        Operation update_op(Optimizer optimizer, Tensor g);
    }
}
