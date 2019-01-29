using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class math_ops
    {
        public static Tensor matmul(Tensor a, Tensor b,
            bool transpose_a = false, bool transpose_b = false,
            bool adjoint_a = false, bool adjoint_b = false,
            bool a_is_sparse = false, bool b_is_sparse = false,
            string name = "")
        {
            Tensor result = null;

            Python.with<ops.name_scope>(new ops.name_scope(name, "MatMul", new Tensor[] { a, b }), scope =>
            {
                name = scope;

                if (transpose_a && adjoint_a)
                    throw new ValueError("Only one of transpose_a and adjoint_a can be True.");
                if (transpose_b && adjoint_b)
                    throw new ValueError("Only one of transpose_b and adjoint_b can be True.");

                a = ops.convert_to_tensor(a, name: "a");
                b = ops.convert_to_tensor(b, name: "b");

                result = gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
            });

            return result;
        }
    }
}
