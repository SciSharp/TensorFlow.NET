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
        /// true if unmanaged buffer has been freed.
        /// </summary>
        private bool deallocator_called => _deallocatorArgs.deallocator_called;

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
        public Tensor(#1[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(#1)), new long[]{data.Length}, data, Marshal.SizeOf<#1>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(#1[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(#1)), shape, data, Marshal.SizeOf<#1>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(#1 value, TF_DataType? dType = null)
        {
            var v = (#1*)Marshal.AllocHGlobal(sizeof(#1));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) => {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(#1)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(#1), deallocator: deallocator, ref _deallocatorArgs);
        }
        %
#else



        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(sbyte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(sbyte)), new long[] { data.Length }, data, Marshal.SizeOf<sbyte>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(sbyte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(sbyte)), shape, data, Marshal.SizeOf<sbyte>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(sbyte value, TF_DataType? dType = null)
        {
            var v = (sbyte*)Marshal.AllocHGlobal(sizeof(sbyte));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(sbyte)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(sbyte), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(byte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(byte)), new long[] { data.Length }, data, Marshal.SizeOf<byte>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(byte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(byte)), shape, data, Marshal.SizeOf<byte>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(byte value, TF_DataType? dType = null)
        {
            var v = (byte*)Marshal.AllocHGlobal(sizeof(byte));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(byte)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(byte), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(short[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(short)), new long[] { data.Length }, data, Marshal.SizeOf<short>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(short[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(short)), shape, data, Marshal.SizeOf<short>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(short value, TF_DataType? dType = null)
        {
            var v = (short*)Marshal.AllocHGlobal(sizeof(short));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(short)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(short), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ushort[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ushort)), new long[] { data.Length }, data, Marshal.SizeOf<ushort>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ushort[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ushort)), shape, data, Marshal.SizeOf<ushort>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ushort value, TF_DataType? dType = null)
        {
            var v = (ushort*)Marshal.AllocHGlobal(sizeof(ushort));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(ushort)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(ushort), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(int[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(int)), new long[] { data.Length }, data, Marshal.SizeOf<int>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(int[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(int)), shape, data, Marshal.SizeOf<int>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(int value, TF_DataType? dType = null)
        {
            var v = (int*)Marshal.AllocHGlobal(sizeof(int));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(int)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(int), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(uint[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(uint)), new long[] { data.Length }, data, Marshal.SizeOf<uint>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(uint[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(uint)), shape, data, Marshal.SizeOf<uint>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(uint value, TF_DataType? dType = null)
        {
            var v = (uint*)Marshal.AllocHGlobal(sizeof(uint));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(uint)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(uint), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(long[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(long)), new long[] { data.Length }, data, Marshal.SizeOf<long>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(long[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(long)), shape, data, Marshal.SizeOf<long>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(long value, TF_DataType? dType = null)
        {
            var v = (long*)Marshal.AllocHGlobal(sizeof(long));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(long)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(long), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ulong[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ulong)), new long[] { data.Length }, data, Marshal.SizeOf<ulong>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ulong[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ulong)), shape, data, Marshal.SizeOf<ulong>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ulong value, TF_DataType? dType = null)
        {
            var v = (ulong*)Marshal.AllocHGlobal(sizeof(ulong));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(ulong)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(ulong), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(float[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(float)), new long[] { data.Length }, data, Marshal.SizeOf<float>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(float[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(float)), shape, data, Marshal.SizeOf<float>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(float value, TF_DataType? dType = null)
        {
            var v = (float*)Marshal.AllocHGlobal(sizeof(float));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(float)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(float), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(double[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(double)), new long[] { data.Length }, data, Marshal.SizeOf<double>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(double[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(double)), shape, data, Marshal.SizeOf<double>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(double value, TF_DataType? dType = null)
        {
            var v = (double*)Marshal.AllocHGlobal(sizeof(double));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(double)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(double), deallocator: deallocator, ref _deallocatorArgs);
        }

        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(Complex[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(Complex)), new long[] { data.Length }, data, Marshal.SizeOf<Complex>());
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(Complex[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(Complex)), shape, data, Marshal.SizeOf<Complex>());
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(Complex value, TF_DataType? dType = null)
        {
            var v = (Complex*)Marshal.AllocHGlobal(sizeof(Complex));
            *v = value;
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs arg) =>
            {
                if (arg.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                arg.deallocator_called = true;
                //_handle = IntPtr.Zero;
            };
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(Complex)), dims: new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(Complex), deallocator: deallocator, ref _deallocatorArgs);
        }
#endif

        /// <summary>
        /// Create a string Tensor from the given string
        /// </summary>
        public unsafe Tensor(string str)
        {
            var buffer = Encoding.UTF8.GetBytes(str);
            var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
            var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(handle);
            Marshal.WriteInt64(tensor, 0);
            fixed (byte* src = &buffer[0])
                c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);
            _handle = handle;
            status.Check(true);
        }

        public Tensor(NDArray nd, TF_DataType? tensorDType = null)
        {
            _handle = Allocate(nd, tensorDType: tensorDType);
        }

        private unsafe IntPtr Allocate(NDArray nd, TF_DataType? tensorDType = null)
        {
            if (tensorDType == TF_DataType.TF_STRING && nd.dtype.Name == "Byte")
            {
                var buffer = nd.Data<byte>();
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
                    return new Tensor(UTF8Encoding.UTF8.GetBytes(nd.Data<string>(0)), TF_DataType.TF_STRING);
                default:
                    throw new NotImplementedException($"Marshal.Copy failed for {nd.dtype.Name}.");
            }

            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref DeallocatorArgs args) =>
            {
                if (args.deallocator_called)
                    return;
                Marshal.FreeHGlobal(values);
                args.deallocator_called = true;
            };

            var tfHandle = c_api.TF_NewTensor(dataType,
                dims,
                dims.Length,
                dotHandle,
                (UIntPtr)buffersize,
                deallocator,
                ref _deallocatorArgs);

            return tfHandle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
            _id = ops.uid();
        }

        private bool _isPinnedArray => _deallocatorArgs.gc_handle != IntPtr.Zero;

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
        protected unsafe IntPtr CreateTensorWithoutCopying(TF_DataType dt, long[] shape, Array data, int element_size)
        {
            if (dt == TF_DataType.TF_STRING && data is byte[])
            {
                var buffer = (byte[])data;
                var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
                var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

                IntPtr tensor = c_api.TF_TensorData(handle);
                Marshal.WriteInt64(tensor, 0);
                fixed (byte* src = &buffer[0])
                    c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);

                status.Check(true);
                return handle;
            }
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
        protected unsafe IntPtr CreateTensorWithoutCopying(TF_DataType dt, long[] shape, Array data, int start, int count, int element_size)
        {
            if (start < 0 || start > data.Length - count)
                throw new ArgumentException($"Array length {data.Length} does not match the given shape {new Shape(shape.Cast<int>().ToArray())}");

            // get a handle to the pinned array which we will pass on to the tensor computation engine to use
            var gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            _deallocatorArgs = new DeallocatorArgs() { gc_handle = GCHandle.ToIntPtr(gcHandle) };
            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr ptr, IntPtr len, ref DeallocatorArgs args) =>
            {
                if (args.deallocator_called || args.gc_handle==IntPtr.Zero)
                    return;
                // note: since the ptr given to tensorflow is just the addr of the pinned object we can not directly free it! we need to free the gcHandle instead
                GCHandle.FromIntPtr(args.gc_handle).Free();
                args.deallocator_called = true;
            };

            if (shape == null || shape.Length == 0)
                return TF_NewTensor(dt, new long[0], 0, gcHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), deallocator, ref _deallocatorArgs);
            else
                return TF_NewTensor(dt, shape, shape.Length, gcHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), deallocator, ref _deallocatorArgs);
        }

        private DeallocatorArgs _deallocatorArgs = new DeallocatorArgs() { gc_handle = IntPtr.Zero };
    }
}
