using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(Tensor x, Tensor y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "add", new Tensor[] { x, y }), scope =>
            {
                return gen_math_ops.add(x, y, scope);
            });
        }

        public static Tensor operator +(Tensor x, int y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "add", new object[] { x, y }), scope =>
            {
                var y1 = ops.convert_to_tensor(y, x.dtype.as_base_dtype(), name: "y");
                return gen_math_ops.add(x, y1, scope);
            });
        }

        public static Tensor operator -(Tensor t1)
        {
            return gen_math_ops.neg(t1);
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
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "mul", new Tensor[] { x, y }), scope =>
            {
                return gen_math_ops.mul(x, y, name: scope);
            });
        }

        public static Tensor operator *(Tensor x, int y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "mul", new object[] { x, y }), scope =>
            {
                var y1 = ops.convert_to_tensor(y, x.dtype.as_base_dtype(), name: "y");
                return gen_math_ops.mul(x, y1, name: scope);
            });
        }

        public static Tensor operator /(Tensor x, Tensor y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("truediv/", "truediv", new Tensor[] { x, y }), scope =>
            {
                return gen_math_ops.real_div(x, y, scope);
            });
        }

        public static Tensor operator /(Tensor x, double y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("truediv/", "truediv", new object[] { x, y }), scope =>
            {
                var y1 = ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y");
                return gen_math_ops.real_div(x, y1, scope);
            });
        }

        public static Tensor operator %(Tensor x, Tensor y)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope("", "mod", new object[] { x, y }), scope =>
            {
                return gen_math_ops.floor_mod(x, y, scope);
            });
        }

        public static Tensor operator >(Tensor x, int y)
        {
            return gen_array_ops.greater(x, y);
        }

        public static Tensor operator <(Tensor x, int y)
        {
            return gen_array_ops.less(x, y);
        }
    }
}
