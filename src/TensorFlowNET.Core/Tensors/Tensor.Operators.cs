/*****************************************************************************
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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Numerics;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(Tensor lhs, ResourceVariable rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, NDArray rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(NDArray lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, sbyte rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(sbyte lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, byte rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(byte lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, short rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(short lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, ushort rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(ushort lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, int rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(int lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, uint rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(uint lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, ulong rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(ulong lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, long rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(long lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, float rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(float lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, double rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(double lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Tensor lhs, Complex rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator +(Complex lhs, Tensor rhs) => BinaryOpWrapper("add", lhs, rhs);
        public static Tensor operator -(Tensor lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, NDArray rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(NDArray lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, sbyte rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(sbyte lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, byte rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(byte lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, short rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(short lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, ushort rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(ushort lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, int rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(int lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, uint rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(uint lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, ulong rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(ulong lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, long rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(long lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, float rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(float lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, double rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(double lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Tensor lhs, Complex rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator -(Complex lhs, Tensor rhs) => BinaryOpWrapper("sub", lhs, rhs);
        public static Tensor operator *(Tensor lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, NDArray rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(NDArray lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, sbyte rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(sbyte lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, byte rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(byte lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, short rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(short lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, ushort rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(ushort lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, int rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(int lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, uint rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(uint lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, ulong rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(ulong lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, long rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(long lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, float rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(float lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, double rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(double lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Tensor lhs, Complex rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator *(Complex lhs, Tensor rhs) => BinaryOpWrapper("mul", lhs, rhs);
        public static Tensor operator /(Tensor lhs, Tensor rhs) => BinaryOpWrapper("truediv", lhs, rhs);
        public static Tensor operator /(Tensor lhs, NDArray rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(NDArray lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, sbyte rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(sbyte lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, byte rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(byte lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, short rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(short lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, ushort rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(ushort lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, int rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(int lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, uint rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(uint lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, ulong rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(ulong lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, long rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(long lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, float rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(float lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, double rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(double lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Tensor lhs, Complex rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator /(Complex lhs, Tensor rhs) => BinaryOpWrapper("div", lhs, rhs);
        public static Tensor operator %(Tensor lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, NDArray rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(NDArray lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, sbyte rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(sbyte lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, byte rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(byte lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, short rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(short lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, ushort rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(ushort lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, int rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(int lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, uint rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(uint lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, ulong rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(ulong lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, long rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(long lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, float rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(float lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, double rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(double lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Tensor lhs, Complex rhs) => BinaryOpWrapper("mod", lhs, rhs);
        public static Tensor operator %(Complex lhs, Tensor rhs) => BinaryOpWrapper("mod", lhs, rhs);

        public static Tensor operator >(Tensor lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, NDArray rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(NDArray lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, sbyte rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(sbyte lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, byte rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(byte lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, short rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(short lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, ushort rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(ushort lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, int rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(int lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, uint rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(uint lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, ulong rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(ulong lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), ops.convert_to_tensor(rhs));
        public static Tensor operator >(Tensor lhs, long rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(long lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, float rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(float lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, double rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(double lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >(Tensor lhs, Complex rhs) => gen_math_ops.greater(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >(Complex lhs, Tensor rhs) => gen_math_ops.greater(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, NDArray rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(NDArray lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, sbyte rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(sbyte lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, byte rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(byte lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, short rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(short lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, ushort rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(ushort lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, int rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(int lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, uint rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(uint lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, ulong rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(ulong lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, long rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(long lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, float rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(float lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, double rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(double lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <(Tensor lhs, Complex rhs) => gen_math_ops.less(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <(Complex lhs, Tensor rhs) => gen_math_ops.less(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, NDArray rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(NDArray lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, sbyte rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(sbyte lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, byte rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(byte lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, short rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(short lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, ushort rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(ushort lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, int rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(int lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, uint rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(uint lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, ulong rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(ulong lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, long rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(long lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, float rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(float lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, double rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(double lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator >=(Tensor lhs, Complex rhs) => gen_math_ops.greater_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator >=(Complex lhs, Tensor rhs) => gen_math_ops.greater_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, NDArray rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(NDArray lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, sbyte rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(sbyte lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, byte rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(byte lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, short rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(short lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, ushort rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(ushort lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, int rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(int lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, uint rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(uint lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, ulong rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(ulong lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, long rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(long lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, float rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(float lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, double rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(double lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator <=(Tensor lhs, Complex rhs) => gen_math_ops.less_equal(lhs, ops.convert_to_tensor(rhs));
        public static Tensor operator <=(Complex lhs, Tensor rhs) => gen_math_ops.less_equal(ops.convert_to_tensor(lhs), rhs);
        public static Tensor operator -(Tensor x) => gen_math_ops.neg(x);


        private static readonly TF_DataType[] _intTfDataTypes = {
            TF_DataType.TF_INT8, TF_DataType.TF_INT16, TF_DataType.TF_INT32, TF_DataType.TF_INT64,
            TF_DataType.TF_QINT8, TF_DataType.TF_QINT16, TF_DataType.TF_QINT32,
            TF_DataType.TF_UINT8, TF_DataType.TF_UINT16, TF_DataType.TF_UINT32, TF_DataType.TF_UINT64
        };

        private static string div_or_truediv<Tx, Ty>(string name, Tx x, Ty y)
        {
            bool is_floating = false;
            var types = new List<bool>();

            if (x is Tensor t1)
                types.add(t1.dtype.is_floating());

            if (y is Tensor t2)
                types.add(t2.dtype.is_floating());

            is_floating = types.Contains(true);

            return is_floating ? "truediv" : name;
        }

        protected static Tensor BinaryOpWrapper<Tx, Ty>(string name, Tx x, Ty y)
        {
            return tf_with(ops.name_scope(null, name, new { x, y }), scope =>
            {
                var dtype = GetBestDType(x, y);
                var x1 = ops.convert_to_tensor(x, name: "x", dtype: dtype);
                var y1 = ops.convert_to_tensor(y, name: "y", dtype: dtype);
                string newname = scope;

                return name.ToLowerInvariant() switch
                {
                    "add" => math_ops.add_v2(x1, y1, name: newname),
                    "div" => math_ops.div(x1, y1, name: newname),
                    "floordiv" => gen_math_ops.floor_div(x1, y1, name: newname),
                    "truediv" => math_ops.truediv(x1, y1, name: newname),
                    "mul" => math_ops.multiply(x1, y1, name: newname),
                    "sub" => gen_math_ops.sub(x1, y1, name: newname),
                    "mod" => gen_math_ops.floor_mod(x1, y1, name: newname),
                    _ => throw new NotImplementedException($"BinaryOpWrapper: {name} - {typeof(Tx).Name}, {typeof(Ty).Name}")
                };
            });
        }

        static TF_DataType GetBestDType<Tx, Ty>(Tx x, Ty y)
        {
            var dtype1 = x.GetDataType();
            var dtype2 = y.GetDataType();
            if (dtype1.is_integer() && dtype2.is_floating())
                return dtype2;
            else if (dtype1.is_floating() && dtype2.is_integer())
                return dtype1;
            else
                return dtype1;
        }
    }
}
