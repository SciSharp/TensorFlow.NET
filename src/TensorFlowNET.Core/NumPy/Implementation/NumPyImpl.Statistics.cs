using System;
using System.Collections.Generic;

namespace Tensorflow.NumPy
{
    public partial class NumPyImpl
    {
        public NDArray average(NDArray a, int axis = -1, NDArray? weights = null, bool returned = false)
        {
            var dtype = NumPyUtils.GetResultType(a.dtype, np.float64);
            if(weights is null)
            {
                var tensorA = math_ops.cast(a, dtype);
                var nd = math_ops.reduce_mean(tensorA, axis);
                return new NDArray(nd);
            }
            else
            {
                var tensorW = math_ops.cast(weights, dtype);
                if(a.rank != weights.rank)
                {
                    var weights_sum = math_ops.reduce_sum(tensorW);
                    var axes = np.array(new[,] { { axis }, { 0 } });
                    var avg = math_ops.tensordot(a, weights, axes) / weights_sum;
                }
                
                throw new NotImplementedException("");
            }
        }
    }
}
