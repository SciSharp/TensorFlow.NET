using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class linalg_ops
    {
        public Tensor eye(int num_rows, 
            int num_columns = -1, 
            TensorShape batch_shape = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            string name = null)
        {
            return tf_with(ops.name_scope(name, default_name: "eye", new { num_rows, num_columns, batch_shape }), scope =>
            {
                if (num_columns == -1)
                    num_columns = num_rows;

                bool is_square = num_columns == num_rows;
                var diag_size = Math.Min(num_rows, num_columns);
                if (batch_shape == null)
                    batch_shape = new TensorShape(new int[0]);
                var diag_shape = batch_shape.dims.concat(new[] { diag_size });

                int[] shape = null;
                if (!is_square)
                    shape = batch_shape.dims.concat(new[] { num_rows, num_columns });

                var diag_ones = array_ops.ones(diag_shape, dtype: dtype);
                if (is_square)
                    return array_ops.matrix_diag(diag_ones);
                else
                {
                    var zero_matrix = array_ops.zeros(shape, dtype: dtype);
                    return array_ops.matrix_set_diag(zero_matrix, diag_ones);
                }
            });
        }
    }
}
