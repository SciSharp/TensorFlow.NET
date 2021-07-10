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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.c_api;

namespace Tensorflow
{
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public partial class Tensor
    {
        /// <summary>
        ///     The handle that was used to allocate this tensor, dependent on <see cref="AllocationType"/>.
        /// </summary>
        protected object AllocationHandle;

        /// <summary>
        ///     True if this Tensor holds data allocated by C#.
        /// </summary>
        public bool IsMemoryOwner => AllocationType >= AllocationType.Marshal;

        /// <summary>
        ///     The allocation method used to create this Tensor.
        /// </summary>
        public AllocationType AllocationType { get; protected set; }

        public IntPtr TensorDataPointer => _handle == IntPtr.Zero ? IntPtr.Zero : TF_TensorData(_handle);

        public Tensor()
        {

        }

        /// <summary>
        ///     Create a Tensor object from an existing TF handle
        /// </summary>
        /// <param name="handle">Handle to a <see cref="Tensor"/> object.</param>
        public Tensor(IntPtr handle)
        {
            _handle = handle;
            //no need to set AllocationType = AllocationType.None;
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} String Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        internal Tensor(Array array, Shape? shape = null)
            => InitTensor(array, shape);

        unsafe void InitTensor(Array array, Shape? shape = null)
        {
            shape = shape ?? array.GetShape();
            var dtype = array.GetType().GetElementType().as_tf_dtype();
            var length = (ulong)(array.Length * dtype.get_datatype_size());

            switch (array)
            {
                case bool[] val:
                    fixed (void* addr = &val[0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case int[] val:
                    fixed (void* addr = &val[0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case int[,] val:
                    fixed (void* addr = &val[0, 0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case long[] val:
                    fixed (void* addr = &val[0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case float[] val:
                    fixed (void* addr = &val[0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case float[,] val:
                    fixed (void* addr = &val[0, 0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case double[] val:
                    fixed (void* addr = &val[0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                case double[,] val:
                    fixed (void* addr = &val[0, 0])
                        _handle = TF_NewTensor(shape, dtype, addr, length);
                    break;
                default:
                    throw new NotImplementedException("");
            }
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
            unsafe
            {
                _handle = TF_NewTensor(dType, dims: shape, num_dims: shape.Length, data: data_ptr, len: (ulong)num_bytes);
                AllocationType = TF_TensorData(_handle) == data_ptr ? AllocationType.FromPointer : AllocationType.Tensorflow;
            }
        }

        public unsafe Tensor(NDArray nd)
            => _handle = TF_NewTensor(nd.shape, nd.dtype.as_tf_dtype(), nd.data.ToPointer(), nd.size * nd.dtypesize);

        #region scala
        public Tensor(bool value) => _handle = TF_NewTensor(value);
        public Tensor(byte value) => _handle = TF_NewTensor(value);
        public Tensor(sbyte value) => _handle = TF_NewTensor(value);
        public Tensor(short value) => _handle = TF_NewTensor(value);
        public Tensor(int value) => _handle = TF_NewTensor(value);
        public Tensor(uint value) => _handle = TF_NewTensor(value);
        public Tensor(long value) => _handle = TF_NewTensor(value);
        public Tensor(ulong value) => _handle = TF_NewTensor(value);
        public Tensor(float value) => _handle = TF_NewTensor(value);
        public Tensor(double value) => _handle = TF_NewTensor(value);
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

        /// <summary>
        ///     Create a string Tensor from the given string
        /// </summary>
        public Tensor(string str)
        {
            _handle = StringTensor(new string[] { str }, TensorShape.Scalar);
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} String Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        public Tensor(string[] strings)
        {
            _handle = StringTensor(strings, new TensorShape(strings.Length));
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} String Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _override_dtype = dtype;
            _id = ops.uid();
        }
    }
}