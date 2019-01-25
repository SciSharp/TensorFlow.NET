using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class RefVariable
    {
        public static Tensor operator +(RefVariable t1, int t2)
        {
            var tensor1 = t1._AsTensor();
            var tensor2 = ops.convert_to_tensor(t2, tensor1.dtype, "y");
            return gen_math_ops.add(tensor1, tensor2);
        }
    }
}
