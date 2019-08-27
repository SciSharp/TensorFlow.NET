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
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using static Tensorflow.c_api;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// true if unmanaged buffer has been freed.
        /// </summary>
        private bool _deallocator_called => _deallocatorArgs.deallocator_called;

        /// <summary>
        /// true if the Tensor was created from a managed array
        /// </summary>
        private bool _isPinnedArray => _deallocatorArgs.gc_handle != IntPtr.Zero;

        /// <summary>
        /// True only if the Tensor object was created in a way that the Tensor object itself allocated memory or pinned a managed object.
        /// False if the Tensor was created from a pointer
        /// </summary>
        public bool IsMemoryOwner { get; private set; }

        /// <summary>
        /// This holds values that are used by the unmanaged deallocator callback
        /// </summary>
        private DeallocatorArgs _deallocatorArgs = new DeallocatorArgs() { gc_handle = IntPtr.Zero };

        // note: they must be assigned to a static variable in order to work as unmanaged callbacks
        private static readonly Deallocator _hGlobalDeallocator = FreeHGlobalMemory;
        private static readonly Deallocator _gcHandleDeallocator = FreeGCHandle;
        private static readonly Deallocator _nothingDeallocator = FreeNothing;

        /// <summary>
        /// Create a Tensor object from an existing TF handle
        /// </summary>
        /// <param name="handle"></param>
        public Tensor(IntPtr handle)
        {
            _handle = handle;
            IsMemoryOwner = false;
        }

        /// <summary>
        /// Create a new Tensor from the given unmanaged memory pointer (which must be allocated, fixed or pinned by the caller)
        /// Note: the caller is responsible for freeing the memory. Calling Dispose on this object will dispose the TensorFlow tensor
        /// but not the memory itself!
        /// </summary>
        /// <param name="ptr">Pointer to unmanaged, fixed or pinned memory which the caller owns</param>
        /// <param name="shape">Tensor shape</param>
        /// <param name="dType">TF data type</param>
        /// <param name="num_bytes">Size of the tensor in memory</param>
        public Tensor(IntPtr ptr, long[] shape, TF_DataType dType, int num_bytes)
        {
            _handle = TF_NewTensor(dType, dims: shape, num_dims: shape.Length, data: ptr, len: (UIntPtr)num_bytes, deallocator: _nothingDeallocator, ref _deallocatorArgs);
            IsMemoryOwner = false;
        }

#if _REGEN
        %types=["sbyte", "bool", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "Complex"]
        %foreach types%
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(#1[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(#1)), new long[]{data.Length}, data, Marshal.SizeOf<#1>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(#1[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(#1)), shape, data, Marshal.SizeOf<#1>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(#1 value, TF_DataType? dType = null)
        {
            var v = (#1*)Marshal.AllocHGlobal(sizeof(#1));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(#1)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(#1), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        %
#else

        
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(sbyte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(sbyte)), new long[]{data.Length}, data, Marshal.SizeOf<sbyte>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(sbyte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(sbyte)), shape, data, Marshal.SizeOf<sbyte>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(sbyte value, TF_DataType? dType = null)
        {
            var v = (sbyte*)Marshal.AllocHGlobal(sizeof(sbyte));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(sbyte)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(sbyte), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(bool[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(bool)), new long[]{data.Length}, data, Marshal.SizeOf<bool>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(bool[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(bool)), shape, data, Marshal.SizeOf<bool>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(bool value, TF_DataType? dType = null)
        {
            var v = (bool*)Marshal.AllocHGlobal(sizeof(bool));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(bool)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(bool), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(byte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(byte)), new long[]{data.Length}, data, Marshal.SizeOf<byte>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(byte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(byte)), shape, data, Marshal.SizeOf<byte>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(byte value, TF_DataType? dType = null)
        {
            var v = (byte*)Marshal.AllocHGlobal(sizeof(byte));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(byte)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(byte), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(short[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(short)), new long[]{data.Length}, data, Marshal.SizeOf<short>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(short[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(short)), shape, data, Marshal.SizeOf<short>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(short value, TF_DataType? dType = null)
        {
            var v = (short*)Marshal.AllocHGlobal(sizeof(short));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(short)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(short), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ushort[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ushort)), new long[]{data.Length}, data, Marshal.SizeOf<ushort>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ushort[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ushort)), shape, data, Marshal.SizeOf<ushort>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ushort value, TF_DataType? dType = null)
        {
            var v = (ushort*)Marshal.AllocHGlobal(sizeof(ushort));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(ushort)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(ushort), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(int[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(int)), new long[]{data.Length}, data, Marshal.SizeOf<int>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(int[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(int)), shape, data, Marshal.SizeOf<int>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(int value, TF_DataType? dType = null)
        {
            var v = (int*)Marshal.AllocHGlobal(sizeof(int));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(int)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(int), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(uint[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(uint)), new long[]{data.Length}, data, Marshal.SizeOf<uint>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(uint[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(uint)), shape, data, Marshal.SizeOf<uint>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(uint value, TF_DataType? dType = null)
        {
            var v = (uint*)Marshal.AllocHGlobal(sizeof(uint));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(uint)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(uint), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(long[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(long)), new long[]{data.Length}, data, Marshal.SizeOf<long>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(long[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(long)), shape, data, Marshal.SizeOf<long>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(long value, TF_DataType? dType = null)
        {
            var v = (long*)Marshal.AllocHGlobal(sizeof(long));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(long)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(long), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ulong[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ulong)), new long[]{data.Length}, data, Marshal.SizeOf<ulong>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ulong[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(ulong)), shape, data, Marshal.SizeOf<ulong>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ulong value, TF_DataType? dType = null)
        {
            var v = (ulong*)Marshal.AllocHGlobal(sizeof(ulong));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(ulong)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(ulong), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(float[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(float)), new long[]{data.Length}, data, Marshal.SizeOf<float>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(float[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(float)), shape, data, Marshal.SizeOf<float>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(float value, TF_DataType? dType = null)
        {
            var v = (float*)Marshal.AllocHGlobal(sizeof(float));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(float)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(float), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(double[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(double)), new long[]{data.Length}, data, Marshal.SizeOf<double>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(double[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(double)), shape, data, Marshal.SizeOf<double>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(double value, TF_DataType? dType = null)
        {
            var v = (double*)Marshal.AllocHGlobal(sizeof(double));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(double)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(double), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
        
        /// <summary>
        /// Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(Complex[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(Complex)), new long[]{data.Length}, data, Marshal.SizeOf<Complex>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(Complex[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorWithoutCopying(dType ?? dtypes.as_dtype(typeof(Complex)), shape, data, Marshal.SizeOf<Complex>());
            IsMemoryOwner=true;
        }

        /// <summary>
        /// Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(Complex value, TF_DataType? dType = null)
        {
            var v = (Complex*)Marshal.AllocHGlobal(sizeof(Complex));
            *v = value;
            _handle = TF_NewTensor(dType ?? dtypes.as_dtype(typeof(Complex)), dims:new long[0], num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof(Complex), deallocator: _hGlobalDeallocator, ref _deallocatorArgs);
            IsMemoryOwner=true;
        }
#endif

        /// <summary>
        /// Create a string Tensor from the given string
        /// </summary>
        public unsafe Tensor(string str)
        {
            var status = new Status();
            var buffer = Encoding.UTF8.GetBytes(str);
            var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
            var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(handle);
            Marshal.WriteInt64(tensor, 0);
            fixed (byte* src = buffer)
                c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);
            _handle = handle;
            status.Check(true);
        }

        public unsafe Tensor(NDArray nd, TF_DataType? tensorDType = null)
        {
            // todo: handle nd of type "String" here too
            if (tensorDType == TF_DataType.TF_STRING && nd.typecode == NPTypeCode.Byte)
            {
                if (nd.Unsafe.Storage.Shape.IsContiguous)
                {
                    var bytesLength = (UIntPtr)nd.size;
                    var size = c_api.TF_StringEncodedSize(bytesLength);
                    var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr) ((ulong) size + 8));

                    IntPtr tensor = c_api.TF_TensorData(handle);
                    Marshal.WriteInt64(tensor, 0);

                    var status = new Status();
                    c_api.TF_StringEncode((byte*) nd.Unsafe.Address, bytesLength, (sbyte*) (tensor + sizeof(Int64)), size, status);

                    status.Check(true);
                    _handle = handle;
                    IsMemoryOwner = false;
                } 
                else
                {
                    var buffer = nd.ToArray<byte>();
                    var size = c_api.TF_StringEncodedSize((UIntPtr) buffer.Length);
                    var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr) ((ulong) size + 8));

                    IntPtr tensor = c_api.TF_TensorData(handle);
                    Marshal.WriteInt64(tensor, 0);

                    var status = new Status();
                    fixed (byte* src = buffer)
                        c_api.TF_StringEncode(src, (UIntPtr) buffer.Length, (sbyte*) (tensor + sizeof(Int64)), size, status);

                    status.Check(true);
                    _handle = handle;
                    IsMemoryOwner = false;
                }

                return;
            }

            _handle = CreateTensorFromNDArray(nd, tensorDType);
            IsMemoryOwner = true;
        }

        private unsafe IntPtr CreateTensorFromNDArray(NDArray nd, TF_DataType? given_dtype)
        {
            if (nd.dtype.Name == "String")
                throw new NotImplementedException("Support for NDArray of type string not implemented yet");
            IArraySlice arraySlice;
            if (nd.Unsafe.Storage.Shape.IsContiguous == false)
            {
                // the memory is NOT contiguous, so we have to copy the view into a contiguous memory block.
                arraySlice = nd.CloneData();
            }
            else
            {
                // the memory is contiguous
                arraySlice = nd.GetData();
            }
            this.Tag = arraySlice; // keep a reference to the memory block to make sure it is not disposed while TF is using it
            var ptr = new IntPtr(arraySlice.Address);
            int num_bytes = (nd.size * nd.dtypesize);
            var dtype = given_dtype ?? nd.dtype.as_dtype();
            var handle = TF_NewTensor(dtype, dims: nd.shape.Select(i=>(long)i).ToArray(), num_dims: nd.ndim, data: ptr, len: (UIntPtr)num_bytes, deallocator: _nothingDeallocator, ref _deallocatorArgs);
            IsMemoryOwner = false;
            return handle;

        }

        public unsafe Tensor(byte[][] buffer, long[] shape)
        {
            int size = 0;
            foreach (var b in buffer)
            {
                size += (int)TF_StringEncodedSize((UIntPtr)b.Length);
            }
            int totalSize = size + buffer.Length * 8;
            ulong offset = 0;
            IntPtr handle = TF_AllocateTensor(TF_DataType.TF_STRING, shape, shape.Length, (UIntPtr)totalSize);

            // Clear offset table
            IntPtr pOffset = TF_TensorData(handle);
            IntPtr dst = pOffset + buffer.Length * 8;
            IntPtr dstLimit = pOffset + totalSize;
            for (int i = 0; i < buffer.Length; i++)
            {
                Marshal.WriteInt64(pOffset, (long)offset);
                using (var status = new Status())
                {
                    fixed (byte* src = &buffer[i][0])
                    {
                        var written = TF_StringEncode(src, (UIntPtr)buffer[i].Length, (sbyte*)dst, (UIntPtr)(dstLimit.ToInt64() - dst.ToInt64()), status);
                        status.Check(true);
                        pOffset += 8;
                        dst += (int)written;
                        offset += written;
                    }
                }
            }

            _handle = handle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _override_dtype = dtype;
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
        [SuppressMessage("ReSharper", "LocalVariableHidesMember")]
        protected unsafe IntPtr CreateTensorWithoutCopying(TF_DataType dt, long[] shape, Array data, int element_size)
        {
            if (dt == TF_DataType.TF_STRING && data is byte[] buffer)
            {
                var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
                var handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

                IntPtr tensor = c_api.TF_TensorData(handle);
                Marshal.WriteInt64(tensor, 0);

                var status = new Status();
                fixed (byte* src = buffer)
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
            if (shape == null || shape.Length == 0)
                return TF_NewTensor(dt, new long[0], 0, gcHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), _gcHandleDeallocator, ref _deallocatorArgs);
            else
                return TF_NewTensor(dt, shape, shape.Length, gcHandle.AddrOfPinnedObject() + start * element_size, (UIntPtr)(count * element_size), _gcHandleDeallocator, ref _deallocatorArgs);
        }

        [MonoPInvokeCallback(typeof(Deallocator))]
        internal static void FreeHGlobalMemory(IntPtr dataPtr, IntPtr len, ref DeallocatorArgs args)
        {
            if (args.deallocator_called)
                return;

            // NumSharp will dispose
            Marshal.FreeHGlobal(dataPtr);
            args.deallocator_called = true;
        }

        [MonoPInvokeCallback(typeof(Deallocator))]
        internal static void FreeGCHandle(IntPtr dataPtr, IntPtr len, ref DeallocatorArgs args)
        {
            if (args.deallocator_called || args.gc_handle == IntPtr.Zero)
                return;
            // note: since the ptr given to tensorflow is just the addr of the pinned object we can not directly free it! we need to free the gcHandle instead
            //args.gchandle.Free();
            args.deallocator_called = true;
        }

        [MonoPInvokeCallback(typeof(Deallocator))]
        internal static void FreeNothing(IntPtr dataPtr, IntPtr len, ref DeallocatorArgs args)
        {
            args.deallocator_called = true;
        }
        
    }

    /// <summary>
    /// This attribute can be applied to callback functions that will be invoked
    /// from unmanaged code to managed code.
    /// </summary>
    /// <remarks>
    /// <code>
    /// [TensorFlow.MonoPInvokeCallback (typeof (BufferReleaseFunc))]
    /// internal static void MyFreeFunc (IntPtr data, IntPtr length){..}
    /// </code>
    /// </remarks>
    public sealed class MonoPInvokeCallbackAttribute : Attribute
    {
        /// <summary>
        /// Use this constructor to annotate the type of the callback function that 
        /// will be invoked from unmanaged code.
        /// </summary>
        /// <param name="t">T.</param>
        public MonoPInvokeCallbackAttribute(Type t) { }
    }

}
