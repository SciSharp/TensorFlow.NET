using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NumPyImpl
    {
        public NDArray eye(int N, int? M = null, int k = 0, TF_DataType dtype = TF_DataType.TF_DOUBLE)
        {
            if (!M.HasValue)
                M = N;

            var diag_len = min(N, M.Value);
            if (k > 0)
            {
                if (N >= M)
                    diag_len -= k;
                else if (N + k > M)
                    diag_len = M.Value - k;
            }
            else
            {
                if (M >= N)
                    diag_len += k;
                else if (M - k > N)
                    diag_len = N + k;
            }

            var diagonal_ = array_ops.ones(new Shape(diag_len), dtype: dtype);
            var tensor = array_ops.matrix_diag(diagonal: diagonal_, num_rows: N, num_cols: M.Value, k: k);
            return new NDArray(tensor);
        }

        public NDArray frombuffer(byte[] bytes, TF_DataType dtype)
        {
            throw new NotImplementedException("");
        }

        public NDArray linspace<T>(T start, T stop, int num = 50, bool endpoint = true, bool retstep = false,
            TF_DataType dtype = TF_DataType.TF_DOUBLE, int axis = 0)
        {
            var start_tensor = array_ops.constant(start, dtype: dtype);
            var stop_tensor = array_ops.constant(stop, dtype: dtype);

            // var step_tensor = array_ops.constant(np.nan);
            Tensor result = null;

            if (endpoint)
            {
                result = math_ops.linspace(start_tensor, stop_tensor, num, axis: axis);
            }
            else
            {
                if (num > 1)
                {
                    var step = (stop_tensor - start_tensor) / num;
                    var new_stop = math_ops.cast(stop_tensor, step.dtype) - step;
                    start_tensor = math_ops.cast(start_tensor, new_stop.dtype);
                    result = math_ops.linspace(start_tensor, new_stop, num, axis: axis);
                }
                else
                    result = math_ops.linspace(start_tensor, stop_tensor, num, axis: axis);
            }

            return new NDArray(result);
        }

        public (NDArray, NDArray) meshgrid<T>(T[] array, bool copy = true, bool sparse = false)
        {
            var tensors = array_ops.meshgrid(array, copy: copy, sparse: sparse);
            return (new NDArray(tensors[0]), new NDArray(tensors[1]));
        }
    }
}
