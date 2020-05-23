using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        [Obsolete("Please use ToArray<T>() instead.", false)]
        public T[] Data<T>() where T : unmanaged
        {
            return ToArray<T>();
        }

        /// <summary>
        ///     
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] ToArray<T>() where T : unmanaged
        {
            //Are the types matching?
            if (typeof(T).as_dtype() == dtype)
            {
                if (NDims == 0 && size == 1)  //is it a scalar?
                {
                    unsafe
                    {
                        return new T[] { *(T*)buffer };
                    }
                }

                //types match, no need to perform cast
                var ret = new T[size];
                unsafe
                {
                    var len = (long)size;
                    fixed (T* dst = ret)
                    {
                        //T can only be unmanaged, I believe it is safe to say that MemoryCopy is valid for all cases this method can be called.
                        var src = (T*)buffer;
                        len *= (long)itemsize;
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }

                return ret;
            }
            else
            {

                //types do not match, need to perform cast
                if (NDims == 0 && size == 1) //is it a scalar?
                {
                    unsafe
                    {
#if _REGEN
                        #region Compute
		                switch (dtype.as_numpy_dtype().GetTypeCode())
		                {
			                %foreach supported_dtypes,supported_dtypes_lowercase%
			                case NPTypeCode.#1: return new T[] {Converts.ChangeType<T>(*(#2*) buffer)};
			                %
			                case NPTypeCode.String: return new T[] {Converts.ChangeType<T>((string)this)};
			                default:
				                throw new NotSupportedException();
		                }
                        #endregion
#else
                        #region Compute
                        switch (dtype.as_numpy_dtype().GetTypeCode())
                        {
                            case NPTypeCode.Boolean: return new T[] { Converts.ChangeType<T>(*(bool*)buffer) };
                            case NPTypeCode.Byte: return new T[] { Converts.ChangeType<T>(*(byte*)buffer) };
                            case NPTypeCode.Int16: return new T[] { Converts.ChangeType<T>(*(short*)buffer) };
                            case NPTypeCode.UInt16: return new T[] { Converts.ChangeType<T>(*(ushort*)buffer) };
                            case NPTypeCode.Int32: return new T[] { Converts.ChangeType<T>(*(int*)buffer) };
                            case NPTypeCode.UInt32: return new T[] { Converts.ChangeType<T>(*(uint*)buffer) };
                            case NPTypeCode.Int64: return new T[] { Converts.ChangeType<T>(*(long*)buffer) };
                            case NPTypeCode.UInt64: return new T[] { Converts.ChangeType<T>(*(ulong*)buffer) };
                            case NPTypeCode.Char: return new T[] { Converts.ChangeType<T>(*(char*)buffer) };
                            case NPTypeCode.Double: return new T[] { Converts.ChangeType<T>(*(double*)buffer) };
                            case NPTypeCode.Single: return new T[] { Converts.ChangeType<T>(*(float*)buffer) };
                            case NPTypeCode.String: return new T[] { Converts.ChangeType<T>((string)this) };
                            default:
                                throw new NotSupportedException();
                        }
                        #endregion
#endif
                    }
                }

                var ret = new T[size];
                unsafe
                {
                    var len = (long)size;
                    fixed (T* dstRet = ret)
                    {
                        T* dst = dstRet; //local stack copy

#if _REGEN
                        #region Compute
		                switch (dtype.as_numpy_dtype().GetTypeCode())
		                {
			                %foreach supported_dtypes,supported_dtypes_lowercase%
			                case NPTypeCode.#1: new UnmanagedMemoryBlock<#2>((#2*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                %
			                default:
				                throw new NotSupportedException();
		                }
                        #endregion
#else
                        #region Compute
                        switch (dtype.as_numpy_dtype().GetTypeCode())
                        {
                            case NPTypeCode.Boolean: new UnmanagedMemoryBlock<bool>((bool*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Byte: new UnmanagedMemoryBlock<byte>((byte*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int16: new UnmanagedMemoryBlock<short>((short*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt16: new UnmanagedMemoryBlock<ushort>((ushort*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int32: new UnmanagedMemoryBlock<int>((int*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt32: new UnmanagedMemoryBlock<uint>((uint*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Int64: new UnmanagedMemoryBlock<long>((long*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.UInt64: new UnmanagedMemoryBlock<ulong>((ulong*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Char: new UnmanagedMemoryBlock<char>((char*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Double: new UnmanagedMemoryBlock<double>((double*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.Single: new UnmanagedMemoryBlock<float>((float*)buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
                            case NPTypeCode.String: throw new NotSupportedException("Unable to convert from string to other dtypes"); //TODO! this should call Converts.To<T> 
                            default:
                                throw new NotSupportedException();
                        }
                        #endregion
#endif

                    }
                }

                return ret;
            }
        }

        /// <summary>
        /// Copy of the contents of this Tensor into a NumPy array or scalar.
        /// </summary>
        /// <returns>
        /// A NumPy array of the same shape and dtype or a NumPy scalar, if this
        /// Tensor has rank 0.
        /// </returns>
        public NDArray numpy()
            => GetNDArray(dtype);

        protected unsafe NDArray GetNDArray(TF_DataType dtype)
        {
            UnmanagedStorage storage;
            switch (dtype)
            {
                case TF_DataType.TF_STRING:
                    return StringData();
                case TF_DataType.TF_INT32:
                    storage = new UnmanagedStorage(NPTypeCode.Int32);
                    break;
                case TF_DataType.TF_INT64:
                    storage = new UnmanagedStorage(NPTypeCode.Int64);
                    break;
                case TF_DataType.TF_FLOAT:
                    storage = new UnmanagedStorage(NPTypeCode.Float);
                    break;
                case TF_DataType.TF_DOUBLE:
                    storage = new UnmanagedStorage(NPTypeCode.Double);
                    break;
                default:
                    return BufferToArray();
            }

            storage.Allocate(new Shape(shape));

            var bytesize = (long)this.bytesize;
            System.Buffer.MemoryCopy(buffer.ToPointer(), storage.Address, bytesize, bytesize);

            return new NDArray(storage);
        }

        /*protected unsafe NDArray GetScalar(TF_DataType dtype)
        {
            switch(dtype)
            {
                case TF_DataType.TF_STRING:
                    return (NDArray)StringData()[0];
                case TF_DataType.TF_INT32:
                    return *(int*)buffer;
                case TF_DataType.TF_FLOAT:
                    return *(float*)buffer;
                case TF_DataType.TF_DOUBLE:
                    return *(double*)buffer;
                default:
                    return BufferToArray();
            }
        }*/

        /// <summary>
        /// Copies the memory of current buffer onto newly allocated array.
        /// </summary>
        /// <returns></returns>
        public unsafe byte[] BufferToArray()
        {
            // ReSharper disable once LocalVariableHidesMember
            var bytesize = (long)this.bytesize;
            var data = new byte[bytesize];
            fixed (byte* dst = data)
                System.Buffer.MemoryCopy(buffer.ToPointer(), dst, bytesize, bytesize);

            return data;
        }

        /// <summary>
        ///     Extracts string array from current Tensor.
        /// </summary>
        /// <exception cref="InvalidOperationException">When <see cref="dtype"/> != TF_DataType.TF_STRING</exception>
        public unsafe string[] StringData()
        {
            if (dtype != TF_DataType.TF_STRING)
                throw new InvalidOperationException($"Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})");

            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            long size = 1;
            foreach (var s in TensorShape.dims)
                size *= s;

            var buffer = new byte[size][];
            var src = c_api.TF_TensorData(_handle);
            var srcLen = (IntPtr)(src.ToInt64() + (long)bytesize);
            src += (int)(size * 8);
            using (var status = new Status())
            {
                for (int i = 0; i < buffer.Length; i++)
                {
                    IntPtr dst = IntPtr.Zero;
                    UIntPtr dstLen = UIntPtr.Zero;
                    var read = c_api.TF_StringDecode((byte*)src, (UIntPtr)(srcLen.ToInt64() - src.ToInt64()), (byte**)&dst, &dstLen, status);
                    status.Check(true);
                    buffer[i] = new byte[(int)dstLen];
                    Marshal.Copy(dst, buffer[i], 0, buffer[i].Length);
                    src += (int)read;
                }
            }

            var _str = new string[buffer.Length];
            for (int i = 0; i < _str.Length; i++)
                _str[i] = Encoding.UTF8.GetString(buffer[i]);

            return _str;
        }
    }
}
