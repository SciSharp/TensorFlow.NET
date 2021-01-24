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
using NumSharp.Utilities;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public partial class Tensor
    {
        public unsafe void CopyTo(NDArray nd)
        {
            if (!nd.Shape.IsContiguous)
                throw new ArgumentException("NDArray has to be contiguous (ndarray.Shape.IsContiguous).");

#if _REGEN
            #region Compute
		    switch (nd.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
				    CopyTo<#2>(new Span<#2>(nd.Unsafe.Address, nd.size*nd.dtypesize));
                    break;
			    }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (nd.typecode)
            {
                case NPTypeCode.Boolean:
                    {
                        CopyTo<bool>(new Span<bool>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Byte:
                    {
                        CopyTo<byte>(new Span<byte>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Int16:
                    {
                        CopyTo<short>(new Span<short>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.UInt16:
                    {
                        CopyTo<ushort>(new Span<ushort>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Int32:
                    {
                        CopyTo<int>(new Span<int>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.UInt32:
                    {
                        CopyTo<uint>(new Span<uint>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Int64:
                    {
                        CopyTo<long>(new Span<long>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.UInt64:
                    {
                        CopyTo<ulong>(new Span<ulong>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Char:
                    {
                        CopyTo<char>(new Span<char>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Double:
                    {
                        CopyTo<double>(new Span<double>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                case NPTypeCode.Single:
                    {
                        CopyTo<float>(new Span<float>(nd.Unsafe.Address, nd.size * nd.dtypesize));
                        break;
                    }
                default:
                    throw new NotSupportedException();
            }

            #endregion
#endif
        }

        public void CopyTo<T>(Span<T> destination) where T : unmanaged
        {
            unsafe
            {
                var len = checked((int)this.size);
                //perform regular CopyTo using Span.CopyTo.
                if (typeof(T).as_dtype() == this.dtype && this.dtype != TF_DataType.TF_STRING) //T can't be a string but tensor can.
                {
                    var src = (T*)this.buffer;
                    var srcSpan = new Span<T>(src, len);
                    srcSpan.CopyTo(destination);

                    return;
                }

                if (len > destination.Length)
                    throw new ArgumentException("Destinion was too short to perform CopyTo.");

                //Perform cast to type <T>.
                fixed (T* dst = destination)
                {
                    switch (this.dtype)
                    {
#if _REGEN
                        %foreach supported_numericals_TF_DataType,supported_numericals,supported_numericals_lowercase%
                        case TF_DataType.#1:
                        {
                            var converter = Converts.FindConverter<#3, T>();
                            var src = (#3*) this.buffer;
                            for (var i = 0; i < len; i++)
                                *(dst + i) = converter(unchecked(*(src + i)));
                            return;
                        }
                        %
#else
                        case TF_DataType.TF_BOOL:
                            {
                                var converter = Converts.FindConverter<bool, T>();
                                var src = (bool*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_UINT8:
                            {
                                var converter = Converts.FindConverter<byte, T>();
                                var src = (byte*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_INT16:
                            {
                                var converter = Converts.FindConverter<short, T>();
                                var src = (short*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_UINT16:
                            {
                                var converter = Converts.FindConverter<ushort, T>();
                                var src = (ushort*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_INT32:
                            {
                                var converter = Converts.FindConverter<int, T>();
                                var src = (int*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_UINT32:
                            {
                                var converter = Converts.FindConverter<uint, T>();
                                var src = (uint*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_INT64:
                            {
                                var converter = Converts.FindConverter<long, T>();
                                var src = (long*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_UINT64:
                            {
                                var converter = Converts.FindConverter<ulong, T>();
                                var src = (ulong*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_DOUBLE:
                            {
                                var converter = Converts.FindConverter<double, T>();
                                var src = (double*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
                        case TF_DataType.TF_FLOAT:
                            {
                                var converter = Converts.FindConverter<float, T>();
                                var src = (float*)this.buffer;
                                for (var i = 0; i < len; i++)
                                    *(dst + i) = converter(unchecked(*(src + i)));
                                return;
                            }
#endif
                        case TF_DataType.TF_STRING:
                            {
                                var src = this.StringData();
                                var culture = CultureInfo.InvariantCulture;

                                //pin to prevent GC from moving the span around.
                                fixed (T* _ = destination)
                                    switch (typeof(T).as_dtype())
                                    {
#if _REGEN
                                    %foreach supported_numericals_TF_DataType,supported_numericals,supported_numericals_lowercase%
                                    case TF_DataType.#1: {
                                        var sdst = (#3*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                        for (var i = 0; i < len; i++)
                                            *(sdst + i) = ((IConvertible)src[i]).To#2(culture);
                                        return;
                                    }
                                    %
#else
                                        case TF_DataType.TF_BOOL:
                                            {
                                                var sdst = (bool*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToBoolean(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_UINT8:
                                            {
                                                var sdst = (byte*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToByte(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_INT16:
                                            {
                                                var sdst = (short*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToInt16(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_UINT16:
                                            {
                                                var sdst = (ushort*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToUInt16(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_INT32:
                                            {
                                                var sdst = (int*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToInt32(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_UINT32:
                                            {
                                                var sdst = (uint*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToUInt32(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_INT64:
                                            {
                                                var sdst = (long*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToInt64(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_UINT64:
                                            {
                                                var sdst = (ulong*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToUInt64(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_DOUBLE:
                                            {
                                                var sdst = (double*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToDouble(culture);
                                                return;
                                            }
                                        case TF_DataType.TF_FLOAT:
                                            {
                                                var sdst = (float*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                                for (var i = 0; i < len; i++)
                                                    *(sdst + i) = ((IConvertible)src[i]).ToSingle(culture);
                                                return;
                                            }
#endif
                                        default:
                                            throw new NotSupportedException();
                                    }
                            }
                        case TF_DataType.TF_COMPLEX64:
                        case TF_DataType.TF_COMPLEX128:
                        default:
                            throw new NotSupportedException();
                    }
                }
            }
        }

        public TensorSpec ToTensorSpec()
            => new TensorSpec(shape, dtype, name);
    }
}