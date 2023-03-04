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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray : Tensor, IEnumerable<NDArray>
    {
        public IntPtr data => TensorDataPointer;

        [AutoNumPy]
        public NDArray reshape(Shape newshape) => new NDArray(tf.reshape(this, newshape));
        [AutoNumPy]
        public NDArray astype(TF_DataType dtype) => new NDArray(math_ops.cast(this, dtype));
        public NDArray ravel() => throw new NotImplementedException("");
        public void shuffle(NDArray nd) => np.random.shuffle(nd);

        public unsafe Array ToMultiDimArray<T>() where T : unmanaged
            => NDArrayConverter.ToMultiDimArray<T>(this);

        public byte[] ToByteArray() => BufferToArray();
        public override string ToString() => NDArrayRender.ToString(this);

        public IEnumerator<NDArray> GetEnumerator()
        {
            for (int i = 0; i < dims[0]; i++)
                yield return this[i];
        }

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();

        public static explicit operator NDArray(Array array)
            => new NDArray(array);
    }
}
