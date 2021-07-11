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
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Text;
using static Tensorflow.c_api;
using static Tensorflow.Binding;

namespace Tensorflow
{
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public partial class Tensor
    {
        public IntPtr TensorDataPointer => _handle == IntPtr.Zero ? IntPtr.Zero : TF_TensorData(_handle);

        public Tensor()
        {
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        /// <summary>
        ///     Create a Tensor object from an existing TF handle
        /// </summary>
        /// <param name="handle">Handle to a <see cref="Tensor"/> object.</param>
        public Tensor(IntPtr handle)
        {
            _handle = handle;
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        /// <summary>
        /// Create a new Tensor from the given unmanaged memory pointer (which must be allocated, fixed or pinned by the caller)
        /// Note: the caller is responsible for freeing the memory. Calling Dispose on this object will dispose the TensorFlow tensor
        /// but not the memory itself!
        /// </summary>
        /// <param name="data_ptr">Pointer to unmanaged, fixed or pinned memory which the caller owns</param>
        /// <param name="shape">Tensor shape</param>
        /// <param name="dType">TF data type</param>
        /// <param name="num_bytes">Size of the tensor in memory</param>
        public Tensor(IntPtr data_ptr, long[] shape, TF_DataType dType, int num_bytes)
        {
            _handle = TF_NewTensor(dType, dims: shape, num_dims: shape.Length, data: data_ptr, len: (ulong)num_bytes);
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        public unsafe Tensor(NDArray nd)
        {
            _handle = TF_NewTensor(nd.shape, nd.dtype, nd.data.ToPointer());
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        #region scala
        public Tensor(bool value) => InitTensor(value);
        public Tensor(byte value) => InitTensor(value);
        public Tensor(sbyte value) => InitTensor(value);
        public Tensor(short value) => InitTensor(value);
        public Tensor(int value) => InitTensor(value);
        public Tensor(uint value) => InitTensor(value);
        public Tensor(long value) => InitTensor(value);
        public Tensor(ulong value) => InitTensor(value);
        public Tensor(float value) => InitTensor(value);
        public Tensor(double value) => InitTensor(value);
        #endregion

        #region 1d array
        public Tensor(bool[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(sbyte[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(byte[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(short[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(ushort[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(int[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(uint[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(long[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(ulong[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(float[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(double[] data, Shape? shape = null) => InitTensor(data, shape);
        public Tensor(Complex[] data, Shape? shape = null) => InitTensor(data, shape);
        #endregion

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _override_dtype = dtype;
            _id = ops.uid();
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        internal Tensor(Shape shape, TF_DataType dtype) => InitTensor(shape, dtype);
        internal Tensor(Array array, Shape? shape = null) => InitTensor(array, shape);
        internal Tensor(string value) => InitTensor(value);

        protected unsafe void InitTensor<T>(T data) where T : unmanaged
        {
            _handle = TF_NewTensor(data);
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected unsafe void InitTensor(Shape shape, TF_DataType dtype)
        {
            _handle = TF_NewTensor(shape, dtype, null);
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected void InitTensor(string value)
        {
            _handle = StringTensor(new[] { value }, TensorShape.Scalar);
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected unsafe void InitTensor(Array array, Shape? shape = null)
        {
            shape = shape ?? array.GetShape();
            var dtype = array.GetType().GetElementType().as_tf_dtype();

            switch (array)
            {
                case bool[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case bool[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case bool[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case bool[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case byte[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case byte[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case byte[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case byte[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case int[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case int[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case int[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case int[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case long[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case long[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case long[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case long[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case float[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case float[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case float[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case float[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case double[] val: fixed (void* addr = &val[0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case double[,] val: fixed (void* addr = &val[0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case double[,,] val: fixed (void* addr = &val[0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case double[,,,] val: fixed (void* addr = &val[0, 0, 0, 0]) _handle = TF_NewTensor(shape, dtype, addr); break;
                case string[] val: _handle = StringTensor(val, shape); break;
                default:
                    throw new NotImplementedException("");
            }

            isCreatedInGraphMode = !tf.executing_eagerly();
        }
    }
}