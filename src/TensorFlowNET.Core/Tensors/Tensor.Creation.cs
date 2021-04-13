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
using NumSharp.Backends.Unmanaged;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.Binding;
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

        public Tensor(int value)
        {
            unsafe
            {
                _handle = TF_NewTensor(tf.int32, dims: null, num_dims: 0, data: &value, len: sizeof(int));
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

        /// <summary>
        /// Create a new Tensor from the given unmanaged memory pointer (which must be allocated, fixed or pinned by the caller)
        /// Note: the caller is responsible for freeing the memory. Calling Dispose on this object will dispose the TensorFlow tensor
        /// but not the memory itself!
        /// </summary>
        /// <param name="data_ptr">Pointer to unmanaged, fixed or pinned memory which the caller owns</param>
        /// <param name="shape">Tensor shape</param>
        /// <param name="dType">TF data type</param>
        /// <param name="num_bytes">Size of the tensor in memory</param>
        public unsafe Tensor(void* data_ptr, long[] shape, TF_DataType dType, int num_bytes)
        {
            _handle = TF_NewTensor(dType, dims: shape, num_dims: shape.Length, data: data_ptr, len: (ulong)num_bytes);
            AllocationType = TF_TensorData(_handle).ToPointer() == data_ptr ? AllocationType.FromPointer : AllocationType.Tensorflow;
        }

#if _REGEN
        %types = ["sbyte", "bool", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "Complex"]
        %foreach types%
        
        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(#1[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(#1)), new long[] {data.Length}, data, #(#1=="Complex"|"Marshal.SizeOf<Complex>()"|"sizeof(#(str(#1)))"));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(#1[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(#1)), shape, data, #(#1=="Complex"|"Marshal.SizeOf<Complex>()"|"sizeof(#(str(#1)))"));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(#1 value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(#1)), dims: new long[0], num_dims: 0, len: (UIntPtr) sizeof(#1));
            *(#1*) TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }
        %
#else

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(sbyte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(sbyte)), new long[] { data.Length }, data, sizeof(sbyte));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(sbyte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(sbyte)), shape, data, sizeof(sbyte));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(sbyte value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(sbyte)), dims: new long[0], num_dims: 0, len: sizeof(sbyte));
            *(sbyte*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(bool[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(bool)), new long[] { data.Length }, data, sizeof(bool));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(bool[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(bool)), shape, data, sizeof(bool));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(bool value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(bool)), dims: new long[0], num_dims: 0, len: sizeof(bool));
            *(bool*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(byte[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(byte)), new long[] { data.Length }, data, sizeof(byte));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(byte[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(byte)), shape, data, sizeof(byte));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(byte value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(byte)), dims: new long[0], num_dims: 0, len: sizeof(byte));
            *(byte*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(short[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(short)), new long[] { data.Length }, data, sizeof(short));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(short[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(short)), shape, data, sizeof(short));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(short value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(short)), dims: new long[0], num_dims: 0, len: sizeof(short));
            *(short*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ushort[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(ushort)), new long[] { data.Length }, data, sizeof(ushort));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ushort[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(ushort)), shape, data, sizeof(ushort));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ushort value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(ushort)), dims: new long[0], num_dims: 0, len: sizeof(ushort));
            *(ushort*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(int[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(int)), new long[] { data.Length }, data, sizeof(int));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(int[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(int)), shape, data, sizeof(int));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(int value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(int)), dims: new long[0], num_dims: 0, len: sizeof(int));
            *(int*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(uint[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(uint)), new long[] { data.Length }, data, sizeof(uint));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(uint[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(uint)), shape, data, sizeof(uint));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(uint value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(uint)), dims: new long[0], num_dims: 0, len: sizeof(uint));
            *(uint*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(long[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(long)), new long[] { data.Length }, data, sizeof(long));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(long[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(long)), shape, data, sizeof(long));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(long value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(long)), dims: new long[0], num_dims: 0, len: sizeof(long));
            *(long*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(ulong[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(ulong)), new long[] { data.Length }, data, sizeof(ulong));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(ulong[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(ulong)), shape, data, sizeof(ulong));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(ulong value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(ulong)), dims: new long[0], num_dims: 0, len: sizeof(ulong));
            *(ulong*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(float[] data)
        {
            _handle = CreateTensorFromArray(TF_DataType.TF_FLOAT, new long[] { data.Length }, data, sizeof(float));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(float[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(float)), shape, data, sizeof(float));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(float value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(float)), dims: new long[0], num_dims: 0, len: sizeof(float));
            *(float*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(double[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(double)), new long[] { data.Length }, data, sizeof(double));
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(double[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(double)), shape, data, sizeof(double));
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(double value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(double)), dims: new long[0], num_dims: 0, len: sizeof(double));
            *(double*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }

        /// <summary>
        ///     Create a 1d Tensor from the given linear array and shape
        /// </summary>
        public Tensor(Complex[] data, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(Complex)), new long[] { data.Length }, data, Marshal.SizeOf<Complex>());
        }

        /// <summary>
        ///     Create a N-dimensional Tensor from the given array
        /// </summary>
        public Tensor(Complex[] data, long[] shape, TF_DataType? dType = null)
        {
            _handle = CreateTensorFromArray(dType ?? dtypes.as_dtype(typeof(Complex)), shape, data, Marshal.SizeOf<Complex>());
        }

        /// <summary>
        ///     Create a scalar Tensor from the given value
        /// </summary>
        public unsafe Tensor(Complex value, TF_DataType? dType = null)
        {
            _handle = TF_AllocateTensor(dType ?? dtypes.as_dtype(typeof(Complex)), dims: new long[0], num_dims: 0, len: (ulong)sizeof(Complex));
            *(Complex*)TF_TensorData(_handle) = value;
            AllocationType = AllocationType.Tensorflow;
        }
#endif

        /// <summary>
        ///     Create a string Tensor from the given string
        /// </summary>
        public unsafe Tensor(string str)
        {
            _handle = StringTensor(new string[] { str }, TensorShape.Scalar);
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} String Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        public unsafe Tensor(string[] strings)
        {
            _handle = StringTensor(strings, new TensorShape(strings.Length));
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} String Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        public unsafe Tensor(NDArray nd, TF_DataType? tensorDType = null)
        {
            if (tensorDType == null)
                tensorDType = nd.dtype.as_dtype();

            // todo: handle nd of type "String" here too
            /*if (tensorDType == TF_DataType.TF_STRING && nd.typecode == NPTypeCode.Byte)
            {
                if (nd.Unsafe.Storage.Shape.IsContiguous)
                {
                    var bytesLength = (ulong)nd.size;
                    var size = bytesLength + 1;
                    var handle = TF_AllocateTensor(TF_DataType.TF_STRING, null, 0, size + 8);
                    AllocationType = AllocationType.Tensorflow;

                    IntPtr tensor = c_api.TF_TensorData(handle);
                    Marshal.WriteInt64(tensor, 0);

                    c_api.TF_StringEncode((byte*)nd.Unsafe.Address, bytesLength, (byte*)(tensor + sizeof(long)), size, tf.Status.Handle);
                    tf.Status.Check(true);
                    _handle = handle;
                }
                else
                {
                    var buffer = nd.ToArray<byte>();
                    var size = (ulong)buffer.Length + 1;
                    var handle = TF_AllocateTensor(TF_DataType.TF_STRING, null, 0, size + 8);
                    AllocationType = AllocationType.Tensorflow;

                    IntPtr tensor = c_api.TF_TensorData(handle);
                    Marshal.WriteInt64(tensor, 0);

                    fixed (byte* src = buffer)
                        c_api.TF_StringEncode(src, (ulong)buffer.Length, (byte*)(tensor + sizeof(Int64)), size, tf.Status.Handle);

                    tf.Status.Check(true);
                    _handle = handle;
                }

                return;
            }*/

            CreateTensorFromNDArray(nd, tensorDType);
#if TRACK_TENSOR_LIFE
            print($"New Tensor 0x{_handle.ToString("x16")} {AllocationType} Data: 0x{TensorDataPointer.ToString("x16")}");
#endif
        }

        private unsafe void CreateTensorFromNDArray(NDArray nd, TF_DataType? given_dtype)
        {
            if (nd.typecode == NPTypeCode.String)
                throw new NotImplementedException("Support for NDArray of type string not implemented yet");

            var arraySlice = nd.Unsafe.Storage.Shape.IsContiguous ? nd.GetData() : nd.CloneData();

            _handle = TF_NewTensor(
                given_dtype ?? nd.dtype.as_dtype(),
                dims: nd.shape.Select(i => (long)i).ToArray(),
                num_dims: nd.ndim,
                data: arraySlice.Address,
                len: (ulong)nd.size * (ulong)nd.dtypesize);

            // if TF decided not to perform copy, hold reference for given NDArray.
            if (TensorDataPointer.ToPointer() == arraySlice.Address)
            {
                AllocationType = AllocationType.FromPointer;
                AllocationHandle = arraySlice;
            }
            else
                AllocationType = AllocationType.Tensorflow;
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
        protected IntPtr CreateTensorFromArray(TF_DataType dt, long[] shape, Array data, int element_size)
        {
            if (dt == TF_DataType.TF_STRING && data is byte[] buffer)
                return StringTensor(new byte[][] { buffer }, TensorShape.Scalar);
            return CreateTensorFromArray(dt, shape, data, 0, data.Length, element_size);
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
        protected IntPtr CreateTensorFromArray(TF_DataType dt, long[] shape, Array data, int start, int count, int element_size)
        {
            if (start < 0 || start > data.Length - count)
                throw new ArgumentException($"Array length {data.Length} does not match the given shape {new Shape(shape.Cast<int>().ToArray())}");

            // get a handle to the pinned array which we will pass on to the tensor computation engine to use
            var gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            var pinnedAddr = gcHandle.AddrOfPinnedObject();

            //call NewTensor
            IntPtr handle;
            if (shape == null || shape.Length == 0)
                handle = TF_NewTensor(dt, new long[0], 0, pinnedAddr + start * element_size, (ulong)(count * element_size));
            else
                handle = TF_NewTensor(dt, shape, shape.Length, pinnedAddr + start * element_size, (ulong)(count * element_size));

            //Figure if TF decided to clone or not.
            if (c_api.TF_TensorData(handle) == pinnedAddr)
            {
                AllocationType = AllocationType.GCHandle;
                AllocationHandle = gcHandle;
            }
            else
            {
                AllocationType = AllocationType.Tensorflow;
                gcHandle.Free();
            }

            return handle;
        }
    }
}