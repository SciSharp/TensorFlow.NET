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

using NumSharp;
using System;
using System.Collections.Generic;
using System.Numerics;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
#if _REGEN
        #region Compute
        %operators = ["add", "sub", "mul", "div", "mod"]
        %operators_sign = ["+", "-", "*", "/", "%"]
        %operators_comparers = [">", "<", ">=", "<="]
        %operators_comparers_names = ["greater", "less", "greater_equal", "less_equal"]

        %possabilities = ["NDArray", "sbyte", "byte", "short", "ushort", "int", "uint", "ulong", "long", "float", "double", "Complex"]
		
        %foreach operators, operators_sign%
        public static Tensor operator #2(Tensor lhs, Tensor rhs) => BinaryOpWrapper("#1", lhs, rhs);
            %foreach possabilities%
        public static Tensor operator #2(Tensor lhs, #101 rhs) => BinaryOpWrapper("#1", lhs, rhs);
        public static Tensor operator #2(#101 lhs, Tensor rhs) => BinaryOpWrapper("#1", lhs, rhs);
            %
        %		

        %foreach operators_comparers_names, operators_comparers %
        public static Tensor operator #2(Tensor lhs, Tensor rhs) => gen_math_ops.#1(lhs, rhs);
            %foreach possabilities%
        public static Tensor operator #2(Tensor lhs, #101 rhs) => gen_math_ops.#1(lhs, rhs);
        public static Tensor operator #2(#101 lhs, Tensor rhs) => gen_math_ops.#1(lhs, rhs);
            %
        %
        public static Tensor operator -(Tensor x) => gen_math_ops.neg(x);
        #endregion
#else
        #region Compute

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
        public static Tensor operator >(Tensor lhs, sbyte rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(sbyte lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, byte rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(byte lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, short rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(short lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, ushort rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(ushort lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, int rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(int lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, uint rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(uint lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, ulong rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(ulong lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, long rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(long lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, float rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(float lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, double rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(double lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Tensor lhs, Complex rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator >(Complex lhs, Tensor rhs) => gen_math_ops.greater(lhs, rhs);
        public static Tensor operator <(Tensor lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, NDArray rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(NDArray lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, sbyte rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(sbyte lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, byte rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(byte lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, short rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(short lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, ushort rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(ushort lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, int rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(int lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, uint rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(uint lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, ulong rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(ulong lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, long rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(long lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, float rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(float lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, double rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(double lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Tensor lhs, Complex rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator <(Complex lhs, Tensor rhs) => gen_math_ops.less(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, NDArray rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(NDArray lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, sbyte rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(sbyte lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, byte rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(byte lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, short rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(short lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, ushort rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(ushort lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, int rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(int lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, uint rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(uint lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, ulong rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(ulong lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, long rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(long lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, float rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(float lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, double rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(double lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Tensor lhs, Complex rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator >=(Complex lhs, Tensor rhs) => gen_math_ops.greater_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, NDArray rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(NDArray lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, sbyte rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(sbyte lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, byte rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(byte lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, short rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(short lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, ushort rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(ushort lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, int rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(int lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, uint rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(uint lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, ulong rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(ulong lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, long rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(long lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, float rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(float lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, double rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(double lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Tensor lhs, Complex rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator <=(Complex lhs, Tensor rhs) => gen_math_ops.less_equal(lhs, rhs);
        public static Tensor operator -(Tensor x) => gen_math_ops.neg(x);
        #endregion
#endif

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

        private static Tensor BinaryOpWrapper<Tx, Ty>(string name, Tx x, Ty y)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;

            if (x is Tensor tl)
            {
                dtype = tl.dtype.as_base_dtype();
            }

            if (y is Tensor tr)
            {
                dtype = tr.dtype.as_base_dtype();
            }

            return tf_with(ops.name_scope(null, name, new { x, y }), scope =>
            {
                Tensor result;
                var x1 = ops.convert_to_tensor(x, dtype: dtype, name: "x");
                var y1 = ops.convert_to_tensor(y, dtype: dtype, name: "y");

                switch (name.ToLowerInvariant())
                {
                    case "add":
                        result = math_ops.add_v2(x1, y1, name: scope);
                        break;
                    case "div":
                        result = math_ops.div(x1, y1, name: scope);
                        break;
                    case "floordiv":
                        result = gen_math_ops.floor_div(x1, y1, name: scope);
                        break;
                    case "truediv":
                        result = math_ops.truediv(x1, y1, name: scope);
                        break;
                    case "mul":
                        result = math_ops.multiply(x1, y1, name: scope);
                        break;
                    case "sub":
                        result = gen_math_ops.sub(x1, y1, name: scope);
                        break;
                    case "mod":
                        result = gen_math_ops.floor_mod(x1, y1, name: scope);
                        break;
                    default:
                        throw new NotImplementedException($"BinaryOpWrapper: {name} - {typeof(Tx).Name}, {typeof(Ty).Name}");
                }

                return result;
            });
        }
    }
}
