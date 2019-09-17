using static Tensorflow.Binding;

namespace Tensorflow.Framework
{
    /// <summary>
    /// Represents a sparse tensor.
    /// </summary>
    public class SparseTensor<T> : CompositeTensor, _TensorLike
    {
        long[,] _indices;
        Tensor indices;

        T[] _values;
        Tensor values;

        int[] _dense_shape;
        Tensor dense_shape;

        public SparseTensor(long[,] indices_, T[] values_, int[] dense_shape_)
        {
            tf_with(ops.name_scope(null, "SparseTensor", new { }), delegate
            {
                indices = ops.convert_to_tensor(
                    indices_, name: "indices", dtype: dtypes.int64);
                values = ops.internal_convert_to_tensor(values_, name: "values");
                dense_shape = ops.convert_to_tensor(
                    dense_shape_, name: "dense_shape", dtype: dtypes.int64);
            });

            _indices = indices_;
            _values = values_;
            _dense_shape = dense_shape_;

            var indices_shape = indices.TensorShape.with_rank(2);
            var values_shape = values.TensorShape.with_rank(1);
            var dense_shape_shape = dense_shape.TensorShape.with_rank(1);

            indices_shape[0].merge_with(values_shape.dims[0]);
            indices_shape[1].merge_with(dense_shape_shape.dims[0]);
        }
    }

    public interface _TensorLike
    {
    }

    public static class sparse_tensor_extension
    {
        public static bool is_sparse(this _TensorLike x)
        {
            return x.GetType().Name.Contains("SparseTensor");
        }
    }
}
