﻿/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Linq;
using static Tensorflow.Python;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(double x, Tensor y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(float x, Tensor y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(int x, Tensor y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(Tensor x, Tensor y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(Tensor x, int y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(Tensor x, float y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(Tensor x, double y) => BinaryOpWrapper("add", x, y);

        public static Tensor operator -(Tensor t1) => gen_math_ops.neg(t1);

        public static Tensor operator -(double x, Tensor y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(float x, Tensor y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(int x, Tensor y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, Tensor y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, int y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, float y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, double y) => BinaryOpWrapper("sub", x, y);

        public static Tensor operator *(float x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(double x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(Tensor x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(Tensor x, int y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(Tensor tensor, bool constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, sbyte constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, byte constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, ushort constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, short constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, uint constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, long constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, ulong constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, float constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(Tensor tensor, double constant) => BinaryOpWrapper("mul", tensor, constant);
        public static Tensor operator *(bool constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(sbyte constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(byte constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(ushort constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(short constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(int constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(uint constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(long constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);
        public static Tensor operator *(ulong constant, Tensor tensor) => BinaryOpWrapper("mul", constant, tensor);

        private static readonly TF_DataType[] _intTfDataTypes = {
            TF_DataType.TF_INT8, TF_DataType.TF_INT16, TF_DataType.TF_INT32, TF_DataType.TF_INT64,
            TF_DataType.TF_QINT8, TF_DataType.TF_QINT16, TF_DataType.TF_QINT32,
            TF_DataType.TF_UINT8, TF_DataType.TF_UINT16, TF_DataType.TF_UINT32, TF_DataType.TF_UINT64
        };
        public static Tensor operator /(double x, Tensor y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(float x, Tensor y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(int x, Tensor y) => BinaryOpWrapper("floordiv", x, y);
        public static Tensor operator /(Tensor x, Tensor y) =>
            _intTfDataTypes.Contains(x._dtype)
                ? BinaryOpWrapper("floordiv", x, y)
                : BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(Tensor x, int y) => BinaryOpWrapper("floordiv", x, y);
        public static Tensor operator /(Tensor x, float y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(Tensor x, double y) => BinaryOpWrapper("truediv", x, y);

        public static Tensor operator %(Tensor x, Tensor y) => BinaryOpWrapper("mod", x, y);

        public static Tensor operator >(double x, Tensor y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(float x, Tensor y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(int x, Tensor y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(Tensor x, Tensor y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(Tensor x, int y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(Tensor x, float y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(Tensor x, double y) => gen_math_ops.greater(x, y);

        public static Tensor operator <(double x, Tensor y) => gen_math_ops.less(x, y);
        public static Tensor operator <(float x, Tensor y) => gen_math_ops.less(x, y);
        public static Tensor operator <(int x, Tensor y) => gen_math_ops.less(x, y);
        public static Tensor operator <(Tensor x, Tensor y) => gen_math_ops.less(x, y);
        public static Tensor operator <(Tensor x, int y) => gen_math_ops.less(x, y);
        public static Tensor operator <(Tensor x, float y) => gen_math_ops.less(x, y);
        public static Tensor operator <(Tensor x, double y) => gen_math_ops.less(x, y);

        public static Tensor operator >=(double x, Tensor y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(float x, Tensor y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(int x, Tensor y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(Tensor x, Tensor y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(Tensor x, int y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(Tensor x, float y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >=(Tensor x, double y) => gen_math_ops.greater_equal(x, y);

        public static Tensor operator <=(int x, Tensor y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(float x, Tensor y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(double x, Tensor y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(Tensor x, Tensor y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(Tensor x, int y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(Tensor x, float y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <=(Tensor x, double y) => gen_math_ops.less_equal(x, y);

        private static Tensor BinaryOpWrapper<Tx, Ty>(string name, Tx x, Ty y)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;
            if (x is Tensor tl)
                dtype = tl.dtype.as_base_dtype();
            if (y is Tensor tr)
                dtype = tr.dtype.as_base_dtype();

            var namescope = ops.name_scope(null, name, new { x, y });
            return with(namescope, scope =>
            {
                Tensor result = null;
                var x1 = ops.convert_to_tensor(x, dtype: dtype, name: "x");
                var y1 = ops.convert_to_tensor(y, dtype: dtype, name: "y");

                switch (name.ToLower())
                {
                    case "add":
                        result = gen_math_ops.add(x1, y1, name: scope);
                        break;
                    case "floordiv":
                        result = gen_math_ops.floor_div(x1, y1, name: scope);
                        break;
                    case "truediv":
                        result = gen_math_ops.real_div(x1, y1, name: scope);
                        break;
                    case "mul":
                        result = gen_math_ops.mul(x1, y1, name: scope);
                        break;
                    case "sub":
                        result = gen_math_ops.sub(x1, y1, name: scope);
                        break;
                    case "mod":
                        result = gen_math_ops.floor_mod(x1, y1, name: scope);
                        break;
                    default:
                        throw new NotImplementedException($"BinaryOpWrapper: {name} - {typeof(Tx).Name}, {typeof(Ty)}");
                }

                return result;
            });

        }
    }
}
