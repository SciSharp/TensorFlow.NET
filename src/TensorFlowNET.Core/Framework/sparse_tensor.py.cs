using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Framework
{
    /// <summary>
    /// Represents a sparse tensor.
    /// </summary>
    public class SparseTensor<T> : CompositeTensor, _TensorLike
    {
        long[,] _indices;
        public Tensor indices;

        T[] _values;
        public Tensor values;

        long[] _dense_shape;
        public Tensor dense_shape;

        TensorShape _shape;
        public TensorShape shape => _shape;

        public TF_DataType dtype => dtypes.as_dtype(typeof(T));

        public SparseTensor(long[,] indices_, T[] values_, long[] dense_shape_)
        {
            tf_with(ops.name_scope(null, "SparseTensor", new { }), delegate
            {
                indices = ops.convert_to_tensor(
                    indices_, name: "indices", dtype: dtypes.int64);
                values = ops.convert_to_tensor(values_, name: "values");
                dense_shape = ops.convert_to_tensor(
                    dense_shape_, name: "dense_shape", dtype: dtypes.int64);
            });

            _indices = indices_;
            _values = values_;
            _dense_shape = dense_shape_;

            var indices_shape = indices.TensorShape.with_rank(2);
            var values_shape = values.TensorShape.with_rank(1);
            var dense_shape_shape = dense_shape.TensorShape.with_rank(1);

            indices_shape["0"].merge_with(values_shape[0]);
            indices_shape["1"].merge_with(dense_shape_shape[0]);

            _shape = new TensorShape(_dense_shape.Select(x => Convert.ToInt32(x)).ToArray());
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
