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
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.c_api;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// if original buffer is free.
        /// </summary>
        private bool deallocator_called;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
        }

#if _REGEN
        %types=["sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "Complex"]
        %foreach types%
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(#1[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(#1)), new long[]{data.Length}, data, Marshal.SizeOf<#1>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(#1[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(#1)), shape, data, Marshal.SizeOf<#1>());
        }

        %
#else
        
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(sbyte[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(sbyte)), new long[]{data.Length}, data, Marshal.SizeOf<sbyte>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(sbyte[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(sbyte)), shape, data, Marshal.SizeOf<sbyte>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(byte[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(byte)), new long[]{data.Length}, data, Marshal.SizeOf<byte>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(byte[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(byte)), shape, data, Marshal.SizeOf<byte>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(short[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(short)), new long[]{data.Length}, data, Marshal.SizeOf<short>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(short[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(short)), shape, data, Marshal.SizeOf<short>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ushort[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(ushort)), new long[]{data.Length}, data, Marshal.SizeOf<ushort>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ushort[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(ushort)), shape, data, Marshal.SizeOf<ushort>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(int[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(int)), new long[]{data.Length}, data, Marshal.SizeOf<int>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(int[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(int)), shape, data, Marshal.SizeOf<int>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(uint[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(uint)), new long[]{data.Length}, data, Marshal.SizeOf<uint>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(uint[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(uint)), shape, data, Marshal.SizeOf<uint>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(long[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(long)), new long[]{data.Length}, data, Marshal.SizeOf<long>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(long[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(long)), shape, data, Marshal.SizeOf<long>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ulong[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(ulong)), new long[]{data.Length}, data, Marshal.SizeOf<ulong>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ulong[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(ulong)), shape, data, Marshal.SizeOf<ulong>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(float[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(float)), new long[]{data.Length}, data, Marshal.SizeOf<float>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(float[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(float)), shape, data, Marshal.SizeOf<float>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(double[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(double)), new long[]{data.Length}, data, Marshal.SizeOf<double>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(double[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(double)), shape, data, Marshal.SizeOf<double>());
        }

        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(Complex[] data)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(Complex)), new long[]{data.Length}, data, Marshal.SizeOf<Complex>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(Complex[] data, long[] shape)
        {
            _handle = CreateTensorWithoutCopying(dtypes.as_dtype(typeof(Complex)), shape, data, Marshal.SizeOf<Complex>());
        }
#endif

        public Tensor(NDArray nd, TF_DataType? tensorDType = null)
        {
            _handle = Allocate(nd, tensorDType: tensorDType);
        }

        private unsafe IntPtr Allocate(NDArray nd, TF_DataType? tensorDType = null)
        {
            if (tensorDType == TF_DataType.TF_STRING && nd.dtype.Name == "Byte")
            {
                var buffer=nd.Data<byte>();
                var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
                var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

                IntPtr tensor = c_api.TF_TensorData(handle);
                Marshal.WriteInt64(tensor, 0);
                fixed (byte* src = &buffer[0])
                    c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);

                status.Check(true);
                return handle;
            }

            IntPtr dotHandle = IntPtr.Zero;
            int buffersize = 0;

            if (nd.dtype.Name != "String")
            {
                buffersize = (nd.size * nd.dtypesize);
                dotHandle = Marshal.AllocHGlobal(buffersize);
            }

            var dataType = ToTFDataType(nd.dtype);
            // shape
            var dims = nd.shape.Select(x => (long)x).ToArray();
            var nd1 = nd.ravel();
            switch (nd.dtype.Name)
            {
                case "Boolean":
                    var boolVals = Array.ConvertAll(nd1.Data<bool>(), x => Convert.ToByte(x));
                    Marshal.Copy(boolVals, 0, dotHandle, nd.size);
                    break;
                case "Int16":
                    Marshal.Copy(nd1.Data<short>(), 0, dotHandle, nd.size);
                    break;
                case "Int32":
                    Marshal.Copy(nd1.Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Int64":
                    Marshal.Copy(nd1.Data<long>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd1.Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd1.Data<double>(), 0, dotHandle, nd.size);
                    break;
                case "Byte":
                    Marshal.Copy(nd1.Data<byte>(), 0, dotHandle, nd.size);
                    break;
                case "String":
                    return new Tensor(UTF8Encoding.UTF8.GetBytes(nd.Data<string>(0)));
                default:
                    throw new NotImplementedException($"Marshal.Copy failed for {nd.dtype.Name}.");
            }
            
            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                Marshal.FreeHGlobal(values);
                closure = true;
            };

            var tfHandle = c_api.TF_NewTensor(dataType,
                dims,
                dims.Length,
                dotHandle,
                (UIntPtr)buffersize,
                deallocator,
                ref deallocator_called);

            return tfHandle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
            _id = ops.uid();
        }

        /// <summary>
        /// Creates a new tensor from the given array without copying memory. The array is pinned down and the pointer passed on.
        /// </summary>
        /// <param name="shape">Represents the tensor shape.</param>
        /// <param name="data">The linear array of data, the data must fit in the tensor with the specified dimensions.</param>
        /// <param name="element_size">The number of bytes in memory of a single array element</param>
        /// <remarks>
        /// Use the FromBuffer method to create a tensor that has the specified dimensions
        /// and is initialized with data from the data array.   The data is copied starting
        /// at the start offset, for count bytes and is laid out into the tensor following the
        /// specified dimensions.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected IntPtr CreateTensorWithoutCopying(TF_DataType dt, long[] shape, Array data, int element_size)
        {
            return CreateTensorWithoutCopying(dt, shape, data, 0, data.Length, element_size);
        }

        /// <summary>
        /// Creates a new tensor from a subsection of the given array without copying memory. The array is pinned down and the pointer passed on.
        /// </summary>
        /// <param name="shape">Represents the tensor shape.</param>
        /// <param name="data">The linear array of data, the data must fit in the tensor with the specified dimensions.</param>
        /// <param name="start">The offset into the provided data array where the data resides.</param>
        /// <param name="count">The number of elements to copy from data.</param>
        /// <param name="element_size">The number of bytes in memory of a single array element</param>
        /// <remarks>
        /// Use the FromBuffer method to create a tensor that has the specified dimensions
        /// and is initialized with data from the data array.   The data is copied starting
        /// at the start offset, for count bytes and is laid out into the tensor following the
        /// specified dimensions.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected IntPtr CreateTensorWithoutCopying(TF_DataType dt, long[] shape, Array data, int start, int count, int element_size)
        {
            if (start < 0 || start > data.Length - count)
                throw new ArgumentException($"Array length {data.Length} does not match the given shape {new Shape(shape.Cast<int>().ToArray())}");
            
            // get a handle to the pinned array which we will pass on to the tensor computation engine to use
            var dataHandle = GCHandle.Alloc(data, GCHandleType.Pinned);

            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                dataHandle.Free();
                closure = true;
            };

            if (shape == null)
                return TF_NewTensor(dt, null, 0, dataHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), deallocator, ref deallocator_called);
            else
                return TF_NewTensor(dt, shape, shape.Length, dataHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), deallocator, ref deallocator_called);
        }
    }
}
