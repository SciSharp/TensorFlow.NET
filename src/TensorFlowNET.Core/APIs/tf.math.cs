using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor add(Tensor a, Tensor b)
        {
            return gen_math_ops.add(a, b);
        }

        public static Tensor sub(Tensor a, Tensor b)
        {
            return gen_math_ops.sub(a, b);
        }

        public static Tensor multiply(Tensor x, Tensor y)
        {
            return gen_math_ops.mul(x, y);
        }

        public static Tensor pow(Tensor x, double y)
        {
            return gen_math_ops.pow(x, y);
        }

        /// <summary>
        /// Computes the sum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor reduce_sum(Tensor input, int[] axis = null)
        {
            Tensor rank;
            using (var namescop = new ops.name_scope<Tensor>("", "Rank", new List<Tensor> { input }))
            {
                string name = namescop;
                rank = gen_array_ops.rank(input, name);
            }
            var s = gen_math_ops.sum(input, rank);
            return gen_math_ops.range(0, s);
        }
    }
}
