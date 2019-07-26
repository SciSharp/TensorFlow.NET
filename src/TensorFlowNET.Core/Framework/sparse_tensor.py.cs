namespace Tensorflow.Framework
{
    public interface _TensorLike
    { }

    public class SparseTensor : CompositeTensor, _TensorLike
    {
        private static Tensor _dense_shape { get; set; }

    }

    public static class sparse_tensor
    {
        public static bool is_sparse(this _TensorLike x)
        {
            return x is SparseTensor;
        }
    }
}
