using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class np_utils
    {
        /// <summary>
        /// Converts a class vector (integers) to binary class matrix.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="num_classes"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public static NDArray to_categorical(NDArray y, int num_classes = -1, TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            var y1 = y.astype(np.int32).ToArray<int>();
            // var input_shape = y.shape[..^1];
            var categorical = np.zeros(((int)y.size, num_classes), dtype: dtype);
            // categorical[np.arange(y.size), y] = 1;
            for (var i = 0; i < (int)y.size; i++)
            {
                categorical[i, y1[i]] = 1.0f;
            }

            return categorical;
        }
    }
}
