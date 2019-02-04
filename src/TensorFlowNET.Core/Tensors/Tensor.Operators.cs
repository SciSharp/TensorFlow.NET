using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(Tensor x, Tensor y)
        {
            Tensor t = null;

            Python.with<ops.name_scope>(new ops.name_scope("", "add", new Tensor[] { x, y }), scope =>
            {
                t = gen_math_ops.add(x, y, scope);
            });

            return t;
        }

        public static Tensor operator -(Tensor t1, Tensor t2)
        {
            return gen_math_ops.sub(t1, t2);
        }

        public static Tensor operator *(double x, Tensor y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "mul", new { x, y }),
                scope =>
                {
                    var x1 = ops.convert_to_tensor(x, y.dtype.as_base_dtype(), name: "x");
                    return gen_math_ops.mul(x1, y, name: scope);
                });
        }

        public static Tensor operator *(Tensor x, Tensor y)
        {
            Tensor t = null;

            Python.with<ops.name_scope>(new ops.name_scope("", "mul", new Tensor[] { x, y }), scope =>
            {
                t = gen_math_ops.mul(x, y, name: scope);
            });

            return t;
        }

        public static Tensor operator /(Tensor t1, Tensor t2)
        {
            return gen_math_ops.real_div(t1, t2);
        }
    }
}
