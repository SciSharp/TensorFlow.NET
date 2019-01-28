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
            string name;
            using (var namescop = new ops.name_scope("", "Rank", new List<Tensor> { input }))
            {
                name = namescop;
                rank = gen_array_ops.rank(input, namescop);
            }

            using (var namescope = new ops.name_scope("range", "Range", new List<Tensor> { 0D, input, 1D }))
            {
                name = namescope;
                var start = ops.convert_to_tensor(0D);
                var limit = ops.convert_to_tensor(input);
                var delta = ops.convert_to_tensor(1D);

                var t = gen_math_ops.range(start, limit, delta, name);
            }
                
            var s = gen_math_ops.sum(input, rank);
            return s;
        }
    }
}
