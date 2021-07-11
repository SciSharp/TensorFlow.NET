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
        public unsafe Tensor(IntPtr data_ptr, Shape shape, TF_DataType dtype)
        {
            _handle = TF_NewTensor(shape, dtype, data: data_ptr.ToPointer());
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        public unsafe Tensor(NDArray nd)
        {
            _handle = TF_NewTensor(nd.shape, nd.dtype, nd.data.ToPointer());
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        #region scala
        public Tensor(bool value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(byte value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(sbyte value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(short value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(int value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(uint value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(long value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(ulong value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(float value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(double value) => InitTensor(new[] { value }, Shape.Scalar);
        public Tensor(string value) => InitTensor(new[] { value }, TensorShape.Scalar);
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

        public Tensor(Shape shape, TF_DataType dtype) => InitTensor(shape, dtype);
        public Tensor(Array array, Shape? shape = null) => InitTensor(array, shape);
        public Tensor(byte[] bytes, TF_DataType dtype) => InitTensor(bytes, dtype);

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _override_dtype = dtype;
            _id = ops.uid();
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected unsafe void InitTensor(Shape shape, TF_DataType dtype)
        {
            _handle = TF_NewTensor(shape, dtype, null);
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected unsafe void InitTensor(byte[] bytes, TF_DataType dtype)
        {
            if (dtype == TF_DataType.TF_STRING)
                _handle = StringTensor(new byte[][] { bytes }, TensorShape.Scalar);
            else
                throw new NotImplementedException("");
            isCreatedInGraphMode = !tf.executing_eagerly();
        }

        protected unsafe void InitTensor(Array array, Shape? shape = null)
        {
            isCreatedInGraphMode = !tf.executing_eagerly();

            shape = shape ?? array.GetShape();
            var dtype = array.GetDataType();

            if (shape.size == 0)
            {
                _handle = TF_NewTensor(shape, dtype, null);
                return;
            }

            _handle = array switch
            {
                bool[] val => InitTensor(val, shape, dtype),
                bool[,] val => InitTensor(val, shape, dtype),
                bool[,,] val => InitTensor(val, shape, dtype),
                bool[,,,] val => InitTensor(val, shape, dtype),
                byte[] val => InitTensor(val, shape, dtype),
                byte[,] val => InitTensor(val, shape, dtype),
                byte[,,] val => InitTensor(val, shape, dtype),
                byte[,,,] val => InitTensor(val, shape, dtype),
                int[] val => InitTensor(val, shape, dtype),
                int[,] val => InitTensor(val, shape, dtype),
                int[,,] val => InitTensor(val, shape, dtype),
                int[,,,] val => InitTensor(val, shape, dtype),
                long[] val => InitTensor(val, shape, dtype),
                long[,] val => InitTensor(val, shape, dtype),
                long[,,] val => InitTensor(val, shape, dtype),
                long[,,,] val => InitTensor(val, shape, dtype),
                float[] val => InitTensor(val, shape, dtype),
                float[,] val => InitTensor(val, shape, dtype),
                float[,,] val => InitTensor(val, shape, dtype),
                float[,,,] val => InitTensor(val, shape, dtype),
                double[] val => InitTensor(val, shape, dtype),
                double[,] val => InitTensor(val, shape, dtype),
                double[,,] val => InitTensor(val, shape, dtype),
                double[,,,] val => InitTensor(val, shape, dtype),
                string[] val => StringTensor(val, shape),
                _ => throw new NotImplementedException("")
            };
        }

        unsafe IntPtr InitTensor<T>(T[] array, Shape shape, TF_DataType dtype) where T : unmanaged
        {
            fixed (T* addr = &array[0])
                return TF_NewTensor(shape, dtype, addr);
        }

        unsafe IntPtr InitTensor<T>(T[,] array, Shape shape, TF_DataType dtype) where T : unmanaged
        {
            fixed (T* addr = &array[0, 0])
                return TF_NewTensor(shape, dtype, addr);
        }

        unsafe IntPtr InitTensor<T>(T[,,] array, Shape shape, TF_DataType dtype) where T : unmanaged
        {
            fixed (T* addr = &array[0, 0, 0])
                return TF_NewTensor(shape, dtype, addr);
        }

        unsafe IntPtr InitTensor<T>(T[,,,] array, Shape shape, TF_DataType dtype) where T : unmanaged
        {
            fixed (T* addr = &array[0, 0, 0, 0])
                return TF_NewTensor(shape, dtype, addr);
        }
    }
}