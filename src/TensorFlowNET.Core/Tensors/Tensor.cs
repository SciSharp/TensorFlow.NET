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
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
using Tensorflow.Framework;
#if SERIALIZABLE
using System.Text.Json.Serialization;
#endif

namespace Tensorflow
{
    /// <summary>
    /// A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
    /// Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.
    /// </summary>
    [SuppressMessage("ReSharper", "ConvertToAutoProperty")]
    public partial class Tensor : DisposableObject, ITensorOrOperation, _TensorLike
    {
        private readonly int _id;
        private readonly Operation _op;
        private readonly int _value_index;
        private TF_Output? _tf_output;
        private readonly TF_DataType _override_dtype;
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public int Id => _id;

        /// <summary>
        ///     The Graph that contains this tensor.
        /// </summary>
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public Graph graph => op?.graph;

        /// <summary>
        ///     The Operation that produces this tensor as an output.
        /// </summary>
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public Operation op => _op;
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public Tensor[] outputs => op.outputs;

        /// <summary>
        ///     The string name of this tensor.
        /// </summary>
        public string name => $"{(op == null ? "<unnamed Operation>" : $"{op.name}:{_value_index}")}";

        /// <summary>
        ///     The index of this tensor in the outputs of its Operation.
        /// </summary>
        public int value_index => _value_index;

        /// <summary>
        ///     The DType of elements in this tensor.
        /// </summary>
        public TF_DataType dtype => _handle == IntPtr.Zero ? _override_dtype : c_api.TF_TensorType(_handle);
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public ulong bytesize => _handle == IntPtr.Zero ? 0 : c_api.TF_TensorByteSize(_handle);
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public ulong itemsize => _handle == IntPtr.Zero ? 0 : c_api.TF_DataTypeSize(dtype);
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public ulong size => _handle == IntPtr.Zero ? 0 : bytesize / itemsize;
        private IntPtr buffer => _handle == IntPtr.Zero ? IntPtr.Zero : c_api.TF_TensorData(_handle);
        public int num_consumers(TF_Output oper_out) => _handle == IntPtr.Zero ? 0 : c_api.TF_OperationOutputNumConsumers(oper_out);
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public int NDims => rank;

        /// <summary>
        ///     The name of the device on which this tensor will be produced, or null.
        /// </summary>
        public string Device => op.Device;
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public int[] dims => shape;

        /// <summary>
        ///     Used for keep other pointer when do implicit operating
        /// </summary>
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public object Tag { get; set; }


        /// <summary>
        ///     Returns the shape of a tensor.
        /// </summary>
        /// <remarks>https://www.tensorflow.org/api_docs/python/tf/shape</remarks>
        public int[] shape
        {
            get
            {
                var dims = new long[rank < 0 ? 0 : rank];

                if (_handle == IntPtr.Zero)
                {
                    using (var status = new Status())
                    {
                        c_api.TF_GraphGetTensorShape(op.graph, _as_tf_output(), dims, rank, status);
                        status.Check();
                    }
                }
                else
                {
                    for (int i = 0; i < rank; i++)
                        dims[i] = c_api.TF_Dim(_handle, i);
                }

                return dims.Select(x => ((IConvertible) x).ToInt32(CultureInfo.InvariantCulture)).ToArray();
            }

            set
            {
                using (var status = new Status())
                {
                    if (value == null)
                        c_api.TF_GraphSetTensorShape(this.graph, this._as_tf_output(), null, -1, status);
                    else
                        c_api.TF_GraphSetTensorShape(this.graph, this._as_tf_output(), value.Select(Convert.ToInt64).ToArray(), value.Length, status);

                    status.Check(true);
                }
            }
        }

        public int[] _shape_tuple()
        {
            return rank < 0 ? null : shape;
        }

#if SERIALIZABLE
        [JsonIgnore]
#endif
        public TensorShape TensorShape => rank < 0 ? new TensorShape() : tensor_util.to_shape(shape);

        /// <summary>
        ///     Updates the shape of this tensor.
        /// </summary>
        public void set_shape(TensorShape shape) 
        {
            this.shape = shape.rank > 0 ? shape.dims : null;
        }

        /// <summary>
        ///     Updates the shape of this tensor.
        /// </summary>
        public void set_shape(Tensor shape)
        {
            // ReSharper disable once MergeConditionalExpression
            this.shape = shape is null ? null : shape.shape;
        }

        /// <summary>
        /// number of dimensions <br></br>
        /// -1 Unknown  <br></br>
        /// 0	Scalar (magnitude only) <br></br>
        /// 1	Vector (magnitude and direction) <br></br>
        /// 2	Matrix (table of numbers) <br></br>
        /// 3	3-Tensor (cube of numbers) <br></br>
        /// n	n-Tensor (you get the idea)
        /// </summary>
        /// <remarks>https://www.tensorflow.org/api_docs/python/tf/rank</remarks>
        public int rank
        {
            get
            {
                if (_handle == IntPtr.Zero)
                {
                    using (var status = new Status())
                    {
                        var output = _as_tf_output();
                        int ndim = c_api.TF_GraphGetTensorNumDims(op.graph, output, status);
                        status.Check();
                        return ndim;
                    }
                }

                return c_api.TF_NumDims(_handle);
            }
        }

        /// <summary>
        ///     Returns a list of Operations that consume this tensor.
        /// </summary>
        /// <returns></returns>
        public Operation[] consumers()
        {
            var output = _as_tf_output();
            var consumer_names = c_api.TF_OperationOutputConsumers_wrapper(output);
            return consumer_names.Select(x => graph.OperationByName(x)).ToArray();
        }

        public TF_Output _as_tf_output()
        {
            if (!_tf_output.HasValue)
                _tf_output = new TF_Output(op, value_index);

            return _tf_output.Value;
        }

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
        /// <exception cref="ArgumentException">When <typeparam name="T"> is string </typeparam></exception>
        public T[] ToArray<T>() where T : unmanaged
        {
            //Are the types matching?
            if (typeof(T).as_dtype() == dtype)
            {
                if (NDims == 0 && size == 1)  //is it a scalar?
                {
                    unsafe
                    {
                        return new T[] {*(T*) buffer};
                    }
                }

                //types match, no need to perform cast
                var ret = new T[size];
                unsafe
                {
                    var len = (long) size;
                    fixed (T* dst = ret)
                    {
                        //T can only be unmanaged, I believe it is safe to say that MemoryCopy is valid for all cases this method can be called.
                        var src = (T*) buffer;
                        len *= ((long) itemsize);
                        System.Buffer.MemoryCopy(src, dst, len, len);
                    }
                }

                return ret;
            } else
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
			                case NPTypeCode.Boolean: return new T[] {Converts.ChangeType<T>(*(bool*) buffer)};
			                case NPTypeCode.Byte: return new T[] {Converts.ChangeType<T>(*(byte*) buffer)};
			                case NPTypeCode.Int16: return new T[] {Converts.ChangeType<T>(*(short*) buffer)};
			                case NPTypeCode.UInt16: return new T[] {Converts.ChangeType<T>(*(ushort*) buffer)};
			                case NPTypeCode.Int32: return new T[] {Converts.ChangeType<T>(*(int*) buffer)};
			                case NPTypeCode.UInt32: return new T[] {Converts.ChangeType<T>(*(uint*) buffer)};
			                case NPTypeCode.Int64: return new T[] {Converts.ChangeType<T>(*(long*) buffer)};
			                case NPTypeCode.UInt64: return new T[] {Converts.ChangeType<T>(*(ulong*) buffer)};
			                case NPTypeCode.Char: return new T[] {Converts.ChangeType<T>(*(char*) buffer)};
			                case NPTypeCode.Double: return new T[] {Converts.ChangeType<T>(*(double*) buffer)};
			                case NPTypeCode.Single: return new T[] {Converts.ChangeType<T>(*(float*) buffer)};
			                case NPTypeCode.String: return new T[] {Converts.ChangeType<T>((string)this)};
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
                    var len = (long) size;
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
			                case NPTypeCode.Boolean: new UnmanagedMemoryBlock<bool>((bool*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Byte: new UnmanagedMemoryBlock<byte>((byte*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Int16: new UnmanagedMemoryBlock<short>((short*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.UInt16: new UnmanagedMemoryBlock<ushort>((ushort*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Int32: new UnmanagedMemoryBlock<int>((int*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.UInt32: new UnmanagedMemoryBlock<uint>((uint*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Int64: new UnmanagedMemoryBlock<long>((long*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.UInt64: new UnmanagedMemoryBlock<ulong>((ulong*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Char: new UnmanagedMemoryBlock<char>((char*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Double: new UnmanagedMemoryBlock<double>((double*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
			                case NPTypeCode.Single: new UnmanagedMemoryBlock<float>((float*) buffer, len).CastTo(new UnmanagedMemoryBlock<T>(dst, len), null, null); break;
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
        ///     Copies the memory of current buffer onto newly allocated array.
        /// </summary>
        /// <returns></returns>
        [Obsolete("Please use set_shape(TensorShape shape) instead.", false)]
        public byte[] Data()
        {
            return BufferToArray();
        }

        /// <summary>
        ///     Copies the memory of current buffer onto newly allocated array.
        /// </summary>
        /// <returns></returns>
        public byte[] BufferToArray()
        {
            unsafe
            {
                // ReSharper disable once LocalVariableHidesMember
                var bytesize = (long) this.bytesize;
                var data = new byte[bytesize];
                fixed (byte* dst = data) 
                    System.Buffer.MemoryCopy(buffer.ToPointer(), dst, bytesize, bytesize);

                return data;
            }
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
            var srcLen = (IntPtr) (src.ToInt64() + (long) bytesize);
            src += (int) (size * 8);
            for (int i = 0; i < buffer.Length; i++)
            {
                using (var status = new Status())
                {
                    IntPtr dst = IntPtr.Zero;
                    UIntPtr dstLen = UIntPtr.Zero;
                    var read = c_api.TF_StringDecode((byte*) src, (UIntPtr) (srcLen.ToInt64() - src.ToInt64()), (byte**) &dst, &dstLen, status);
                    status.Check(true);
                    buffer[i] = new byte[(int) dstLen];
                    Marshal.Copy(dst, buffer[i], 0, buffer[i].Length);
                    src += (int) read;
                }
            }

            var _str = new string[buffer.Length];
            for (int i = 0; i < _str.Length; i++)
                _str[i] = Encoding.UTF8.GetString(buffer[i]);

            return _str;
        }

        public Tensor MaybeMove()
        {
            var tensor = c_api.TF_TensorMaybeMove(_handle);
            return tensor;
        }

        /// <summary>
        ///     Evaluates this tensor in a `Session`.
        /// </summary>
        /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
        /// <returns>A <see cref="NumSharp"/> array corresponding to the value of this tensor.</returns>
        public NDArray eval(params FeedItem[] feed_dict)
        {
            return ops._eval_using_default_session(this, feed_dict, graph);
        }

        /// <summary>
        ///     Evaluates this tensor in a `Session`.
        /// </summary>
        /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
        /// <param name="session">The `Session` to be used to evaluate this tensor.</param>
        /// <returns>A <see cref="NumSharp"/> array corresponding to the value of this tensor.</returns>
        public NDArray eval(Session session, params FeedItem[] feed_dict)
        {
            return ops._eval_using_default_session(this, feed_dict, graph, session);
        }

        public override string ToString()
        {
            // this can throw IndexOutOfRangeException 
            switch (rank)
            {
                case -1:
                    return $"tf.Tensor '{name}' shape=<unknown> dtype={dtype}";
                case 0:
                    return $"tf.Tensor '{name}' shape=() dtype={dtype}";
                default:
                    return $"tf.Tensor '{name}' shape=({string.Join(",", shape)}) dtype={dtype}";
            }
        }

        /// <summary>
        ///     Dispose any managed resources.
        /// </summary>
        /// <remarks>Equivalent to what you would perform inside <see cref="DisposableObject.Dispose"/></remarks>
        protected override void DisposeManagedResources()
        {
            AllocationReferenceHolder = null;
        }

        [SuppressMessage("ReSharper", "ConvertIfStatementToSwitchStatement")]
        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            c_api.TF_DeleteTensor(handle);

            if (AllocationHandle == null) 
                return;

            if (AllocationType == AllocationType.GCHandle)
            {
                ((GCHandle) AllocationHandle).Free();
                AllocationHandle = null;
                AllocationType = AllocationType.None;
            } else if (AllocationType == AllocationType.Marshal)
            {
                Marshal.FreeHGlobal((IntPtr) AllocationHandle);
                AllocationHandle = null;
                AllocationType = AllocationType.None;
            } else
                throw new InvalidOperationException($"Tensor.AllocationHandle is not null ({AllocationHandle}) but AllocationType is not matched to a C# allocation type ({AllocationType}).");
        }
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public bool IsDisposed => _disposed;

        // public int tensor_int_val { get; set; }
    }
}