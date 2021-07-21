/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray : Tensor
    {
        public IntPtr data => TensorDataPointer;

        public NDArray[] GetNDArrays()
            => throw new NotImplementedException("");

        public ValueType GetValue(params int[] indices)
            => throw new NotImplementedException("");

        public NDIterator<T> AsIterator<T>(bool autoreset = false) where T : unmanaged
            => throw new NotImplementedException("");

        public bool HasNext() => throw new NotImplementedException("");
        public T MoveNext<T>() => throw new NotImplementedException("");
        [AutoNumPy]
        public NDArray reshape(Shape newshape) => new NDArray(tf.reshape(this, newshape));
        public NDArray astype(TF_DataType dtype) => new NDArray(math_ops.cast(this, dtype));
        public NDArray ravel() => throw new NotImplementedException("");
        public void shuffle(NDArray nd) => throw new NotImplementedException("");
        public Array ToMuliDimArray<T>() => throw new NotImplementedException("");
        public byte[] ToByteArray() => BufferToArray();
        public static string[] AsStringArray(NDArray arr) => throw new NotImplementedException("");

        public override string ToString() => NDArrayRender.ToString(this);
    }
}
