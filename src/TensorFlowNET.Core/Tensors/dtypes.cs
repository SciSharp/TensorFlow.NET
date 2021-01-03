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
using System.Numerics;

namespace Tensorflow
{
    public static class dtypes
    {
        public static TF_DataType @bool = TF_DataType.TF_BOOL;
        public static TF_DataType int8 = TF_DataType.TF_INT8;
        public static TF_DataType int32 = TF_DataType.TF_INT32;
        public static TF_DataType int64 = TF_DataType.TF_INT64;
        public static TF_DataType uint8 = TF_DataType.TF_UINT8;
        public static TF_DataType uint32 = TF_DataType.TF_UINT32;
        public static TF_DataType uint64 = TF_DataType.TF_UINT64;
        public static TF_DataType float32 = TF_DataType.TF_FLOAT; // is that float32?
        public static TF_DataType float16 = TF_DataType.TF_HALF;
        public static TF_DataType float64 = TF_DataType.TF_DOUBLE;
        public static TF_DataType complex = TF_DataType.TF_COMPLEX;
        public static TF_DataType complex64 = TF_DataType.TF_COMPLEX64;
        public static TF_DataType complex128 = TF_DataType.TF_COMPLEX128;
        public static TF_DataType variant = TF_DataType.TF_VARIANT;
        public static TF_DataType resource = TF_DataType.TF_RESOURCE;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns><see cref="System.Type"/> equivalent to <paramref name="type"/>, if none exists, returns null.</returns>
        public static Type as_numpy_dtype(this TF_DataType type)
        {
            switch (type.as_base_dtype())
            {
                case TF_DataType.TF_BOOL:
                    return typeof(bool);
                case TF_DataType.TF_UINT8:
                    return typeof(byte);
                case TF_DataType.TF_INT8:
                    return typeof(sbyte);
                case TF_DataType.TF_INT64:
                    return typeof(long);
                case TF_DataType.TF_UINT64:
                    return typeof(ulong);
                case TF_DataType.TF_INT32:
                    return typeof(int);
                case TF_DataType.TF_UINT32:
                    return typeof(uint);
                case TF_DataType.TF_INT16:
                    return typeof(short);
                case TF_DataType.TF_UINT16:
                    return typeof(ushort);
                case TF_DataType.TF_FLOAT:
                    return typeof(float);
                case TF_DataType.TF_DOUBLE:
                    return typeof(double);
                case TF_DataType.TF_STRING:
                    return typeof(string);
                case TF_DataType.TF_COMPLEX128:
                case TF_DataType.TF_COMPLEX64: //64 is also TF_COMPLEX
                    return typeof(Complex);
                default:
                    return null;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">When <paramref name="type"/> has no equivalent <see cref="NPTypeCode"/></exception>
        public static NPTypeCode as_numpy_typecode(this TF_DataType type)
        {
            switch (type)
            {
                case TF_DataType.TF_BOOL:
                    return NPTypeCode.Boolean;
                case TF_DataType.TF_UINT8:
                    return NPTypeCode.Byte;
                case TF_DataType.TF_INT64:
                    return NPTypeCode.Int64;
                case TF_DataType.TF_INT32:
                    return NPTypeCode.Int32;
                case TF_DataType.TF_INT16:
                    return NPTypeCode.Int16;
                case TF_DataType.TF_UINT64:
                    return NPTypeCode.UInt64;
                case TF_DataType.TF_UINT32:
                    return NPTypeCode.UInt32;
                case TF_DataType.TF_UINT16:
                    return NPTypeCode.UInt16;
                case TF_DataType.TF_FLOAT:
                    return NPTypeCode.Single;
                case TF_DataType.TF_DOUBLE:
                    return NPTypeCode.Double;
                case TF_DataType.TF_STRING:
                    return NPTypeCode.String;
                case TF_DataType.TF_COMPLEX128:
                case TF_DataType.TF_COMPLEX64: //64 is also TF_COMPLEX
                    return NPTypeCode.Complex;
                default:
                    throw new NotSupportedException($"Unable to convert {type} to a NumSharp typecode.");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">When <paramref name="type"/> has no equivalent <see cref="TF_DataType"/></exception>
        public static TF_DataType as_dtype(this Type type, TF_DataType? dtype = null)
        {
            switch (type.Name)
            {
                case "Char":
                    dtype = dtype ?? TF_DataType.TF_UINT8;
                    break;
                case "SByte":
                    dtype = TF_DataType.TF_INT8;
                    break;
                case "Byte":
                    dtype = dtype ?? TF_DataType.TF_UINT8;
                    break;
                case "Int16":
                    dtype = TF_DataType.TF_INT16;
                    break;
                case "UInt16":
                    dtype = TF_DataType.TF_UINT16;
                    break;
                case "Int32":
                    dtype = TF_DataType.TF_INT32;
                    break;
                case "UInt32":
                    dtype = TF_DataType.TF_UINT32;
                    break;
                case "Int64":
                    dtype = TF_DataType.TF_INT64;
                    break;
                case "UInt64":
                    dtype = TF_DataType.TF_UINT64;
                    break;
                case "Single":
                    dtype = TF_DataType.TF_FLOAT;
                    break;
                case "Double":
                    dtype = TF_DataType.TF_DOUBLE;
                    break;
                case "Complex":
                    dtype = TF_DataType.TF_COMPLEX128;
                    break;
                case "String":
                    dtype = TF_DataType.TF_STRING;
                    break;
                case "Boolean":
                    dtype = TF_DataType.TF_BOOL;
                    break;
                default:
                    throw new NotSupportedException($"Unable to convert {type} to a NumSharp typecode.");
            }

            return dtype.Value;
        }

        public static DataType as_datatype_enum(this TF_DataType type)
        {
            return (DataType)type;
        }

        public static TF_DataType as_base_dtype(this TF_DataType type)
        {
            return (int)type > 100 ? (TF_DataType)((int)type - 100) : type;
        }

        public static int name(this TF_DataType type)
        {
            return (int)type;
        }

        public static string as_numpy_name(this TF_DataType type)
            => type switch
            {
                TF_DataType.TF_STRING => "string",
                TF_DataType.TF_UINT8 => "uint8",
                TF_DataType.TF_INT32 => "int32",
                TF_DataType.TF_INT64 => "int64",
                TF_DataType.TF_FLOAT => "float32",
                TF_DataType.TF_BOOL => "bool",
                TF_DataType.TF_RESOURCE => "resource",
                TF_DataType.TF_VARIANT => "variant",
                _ => type.ToString()
            };

        public static Type as_numpy_dtype(this DataType type)
        {
            return type.as_tf_dtype().as_numpy_dtype();
        }

        public static DataType as_base_dtype(this DataType type)
        {
            return (int)type > 100 ? (DataType)((int)type - 100) : type;
        }

        public static TF_DataType as_tf_dtype(this DataType type)
        {
            return (TF_DataType)type;
        }

        public static TF_DataType as_ref(this TF_DataType type)
        {
            return (int)type < 100 ? (TF_DataType)((int)type + 100) : type;
        }

        public static long min(this TF_DataType type)
        {
            throw new NotImplementedException($"min {type.name()}");
        }

        public static long max(this TF_DataType type)
        {
            switch (type)
            {
                case TF_DataType.TF_INT8:
                    return sbyte.MaxValue;
                case TF_DataType.TF_INT16:
                    return short.MaxValue;
                case TF_DataType.TF_INT32:
                    return int.MaxValue;
                case TF_DataType.TF_INT64:
                    return long.MaxValue;
                case TF_DataType.TF_UINT8:
                    return byte.MaxValue;
                case TF_DataType.TF_UINT16:
                    return ushort.MaxValue;
                case TF_DataType.TF_UINT32:
                    return uint.MaxValue;
                default:
                    throw new NotImplementedException($"max {type.name()}");
            }
        }

        public static bool is_complex(this TF_DataType type)
        {
            return type == TF_DataType.TF_COMPLEX || type == TF_DataType.TF_COMPLEX64 || type == TF_DataType.TF_COMPLEX128;
        }

        public static bool is_integer(this TF_DataType type)
        {
            return type == TF_DataType.TF_INT8 || type == TF_DataType.TF_INT16 || type == TF_DataType.TF_INT32 || type == TF_DataType.TF_INT64 ||
                type == TF_DataType.TF_UINT8 || type == TF_DataType.TF_UINT16 || type == TF_DataType.TF_UINT32 || type == TF_DataType.TF_UINT64 ||
                type == TF_DataType.DtInt32Ref || type == TF_DataType.DtInt64Ref;
        }

        public static bool is_floating(this TF_DataType type)
        {
            return type == TF_DataType.TF_HALF || type == TF_DataType.TF_FLOAT || type == TF_DataType.TF_DOUBLE;
        }

        public static bool is_ref_dtype(this TF_DataType type)
        {
            return (int)type > 100;
        }

        public static bool is_compatible_with(this TF_DataType self, TF_DataType other)
        {
            return self.as_datatype_enum() == other.as_datatype_enum();
        }

        public static TF_DataType real_dtype(this TF_DataType self)
        {
            TF_DataType base_ = self.as_base_dtype();
            if (base_ == complex64)
                return float32;
            else if (base_ == complex128)
                return float64;
            else
                return self;
        }

        public static bool is_value_dtype(this TF_DataType type)
        {
            return ((int)type >= 1 && (int)type <= 19)
                || type == TF_DataType.TF_UINT32
                || type == TF_DataType.TF_UINT64;
        }
    }
}
