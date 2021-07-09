using Tensorflow.NumPy;
using System;
using System.Threading.Tasks;
using Tensorflow.Util;

namespace Tensorflow
{
    /// <summary>
    ///     Provides various methods to conversion between types and <see cref="Tensor"/>.
    /// </summary>
    public static class TensorConverter
    {
        /// <summary>
        ///     Convert given <see cref="Array"/> to <see cref="Tensor"/>.
        /// </summary>
        /// <param name="nd">The ndarray to convert, can be regular, jagged or multi-dim array.</param>
        /// <param name="astype">Convert <see cref="Array"/> to given <paramref name="astype"/> before inserting it into a <see cref="Tensor"/>.</param>
        /// <exception cref="NotSupportedException"></exception>
        public static Tensor ToTensor(NDArray nd, TF_DataType? astype = null)
        {
            // return new Tensor(astype == null ? nd : nd.astype(astype.Value.as_numpy_typecode(), false));
            throw new NotImplementedException("");
        }

        /// <summary>
        ///     Convert given <see cref="NDArray"/> to <see cref="Tensor"/>.
        /// </summary>
        /// <param name="nd">The ndarray to convert.</param>
        /// <param name="astype">Convert <see cref="Array"/> to given <paramref name="astype"/> before inserting it into a <see cref="Tensor"/>.</param>
        /// <exception cref="NotSupportedException"></exception>
        public static Tensor ToTensor(NDArray nd, NumpyDType? astype = null)
        {
            // return new Tensor(astype == null ? nd : nd.astype(astype.Value, false));
            throw new NotImplementedException("");
        }

        /// <summary>
        ///     Convert given <see cref="Array"/> to <see cref="Tensor"/>.
        /// </summary>
        /// <param name="array">The array to convert, can be regular, jagged or multi-dim array.</param>
        /// <param name="astype">Convert <see cref="Array"/> to given <paramref name="astype"/> before inserting it into a <see cref="Tensor"/>.</param>
        /// <exception cref="NotSupportedException"></exception>
        public static Tensor ToTensor(Array array, TF_DataType? astype = null)
        {
            if (array == null) throw new ArgumentNullException(nameof(array));
            var arrtype = array.ResolveElementType();

            var astype_type = astype?.as_system_dtype() ?? arrtype;
            if (astype_type == arrtype)
            {
                //no conversion required
                if (astype == TF_DataType.TF_STRING)
                {
                    throw new NotSupportedException(); //TODO! when string is fully implemented.
                }

                if (astype == TF_DataType.TF_INT8)
                {
                    // if (array.Rank != 1 || array.GetType().GetElementType()?.IsArray == true) //is multidim or jagged
                        // array = Arrays.Flatten(array);

                    return new Tensor((sbyte[])array);
                }

                //is multidim or jagged, if so - use NDArrays constructor as it records shape.
                if (array.Rank != 1 || array.GetType().GetElementType().IsArray)
                    return new Tensor(new NDArray(array));

                switch (arrtype.GetTypeCode())
                {
                    case NumpyDType.Boolean: return new Tensor((bool[])array);
                    case NumpyDType.Byte: return new Tensor((byte[])array);
                    case NumpyDType.Int16: return new Tensor((short[])array);
                    case NumpyDType.UInt16: return new Tensor((ushort[])array);
                    case NumpyDType.Int32: return new Tensor((int[])array);
                    case NumpyDType.UInt32: return new Tensor((uint[])array);
                    case NumpyDType.Int64: return new Tensor((long[])array);
                    case NumpyDType.UInt64: return new Tensor((ulong[])array);
                    // case NPTypeCode.Char: return new Tensor((char[])array);
                    case NumpyDType.Double: return new Tensor((double[])array);
                    case NumpyDType.Single: return new Tensor((float[])array);
                    default:
                        throw new NotSupportedException();
                }
            }
            else
            {
                //conversion is required.
                //by this point astype is not null.

                //flatten if required
                /*if (array.Rank != 1 || array.GetType().GetElementType()?.IsArray == true) //is multidim or jagged
                    array = Arrays.Flatten(array);

                try
                {
                    return ToTensor(
                        ArrayConvert.To(array, astype.Value.as_numpy_typecode()),
                        null
                    );
                }
                catch (NotSupportedException)
                {
                    //handle dtypes not supported by ArrayConvert
                    var ret = Array.CreateInstance(astype_type, array.LongLength);
                    Parallel.For(0, ret.LongLength, i => ret.SetValue(Convert.ChangeType(array.GetValue(i), astype_type), i));
                    return ToTensor(ret, null);
                }*/
                throw new NotImplementedException("");
            }
        }

        /// <summary>
        ///     Convert given <see cref="Array"/> to <see cref="Tensor"/>.
        /// </summary>
        /// <param name="constant">The constant scalar to convert</param>
        /// <param name="astype">Convert <paramref name="constant"/> to given <paramref name="astype"/> before inserting it into a <see cref="Tensor"/>.</param>
        /// <exception cref="NotSupportedException"></exception>
        public static Tensor ToTensor<T>(T constant, TF_DataType? astype = null) where T : unmanaged
        {
            //was conversion requested?
            if (astype == null)
            {
                //No conversion required
                var constantType = typeof(T).as_tf_dtype();
                if (constantType == TF_DataType.TF_INT8)
                    return new Tensor((sbyte)(object)constant);

                if (constantType == TF_DataType.TF_STRING)
                    return new Tensor((string)(object)constant);

                /*switch (InfoOf<T>.NPTypeCode)
                {
                    case NPTypeCode.Boolean: return new Tensor((bool)(object)constant);
                    case NPTypeCode.Byte: return new Tensor((byte)(object)constant);
                    case NPTypeCode.Int16: return new Tensor((short)(object)constant);
                    case NPTypeCode.UInt16: return new Tensor((ushort)(object)constant);
                    case NPTypeCode.Int32: return new Tensor((int)(object)constant);
                    case NPTypeCode.UInt32: return new Tensor((uint)(object)constant);
                    case NPTypeCode.Int64: return new Tensor((long)(object)constant);
                    case NPTypeCode.UInt64: return new Tensor((ulong)(object)constant);
                    // case NPTypeCode.Char: return new Tensor(Converts.ToByte(constant));
                    case NPTypeCode.Double: return new Tensor((double)(object)constant);
                    case NPTypeCode.Single: return new Tensor((float)(object)constant);
                    default:
                        throw new NotSupportedException();
                }*/
                throw new NotImplementedException("");
            }

            //conversion required

            /*if (astype == TF_DataType.TF_INT8)
                return new Tensor(Converts.ToSByte(constant));

            if (astype == TF_DataType.TF_STRING)
                return new Tensor(Converts.ToString(constant));

            var astype_np = astype?.as_numpy_typecode();


            switch (astype_np)
            {
                case NPTypeCode.Boolean: return new Tensor(Converts.ToBoolean(constant));
                case NPTypeCode.Byte: return new Tensor(Converts.ToByte(constant));
                case NPTypeCode.Int16: return new Tensor(Converts.ToInt16(constant));
                case NPTypeCode.UInt16: return new Tensor(Converts.ToUInt16(constant));
                case NPTypeCode.Int32: return new Tensor(Converts.ToInt32(constant));
                case NPTypeCode.UInt32: return new Tensor(Converts.ToUInt32(constant));
                case NPTypeCode.Int64: return new Tensor(Converts.ToInt64(constant));
                case NPTypeCode.UInt64: return new Tensor(Converts.ToUInt64(constant));
                case NPTypeCode.Char: return new Tensor(Converts.ToByte(constant));
                case NPTypeCode.Double: return new Tensor(Converts.ToDouble(constant));
                case NPTypeCode.Single: return new Tensor(Converts.ToSingle(constant));
                default:
                    throw new NotSupportedException();
            }*/
            throw new NotImplementedException("");

        }

        /// <summary>
        ///     Convert given <see cref="Array"/> to <see cref="Tensor"/>.
        /// </summary>
        /// <param name="constant">The constant scalar to convert</param>
        /// <param name="astype">Convert <paramref name="constant"/> to given <paramref name="astype"/> before inserting it into a <see cref="Tensor"/>.</param>
        /// <exception cref="NotSupportedException"></exception>
        public static Tensor ToTensor(string constant, TF_DataType? astype = null)
        {
            /*switch (astype)
            {
                //was conversion requested?
                case null:
                case TF_DataType.TF_STRING:
                    return new Tensor(constant);
                //conversion required
                case TF_DataType.TF_INT8:
                    return new Tensor(Converts.ToSByte(constant));
                default:
                    {
                        var astype_np = astype?.as_numpy_typecode();

                        switch (astype_np)
                        {
                            case NPTypeCode.Boolean: return new Tensor(Converts.ToBoolean(constant));
                            case NPTypeCode.Byte: return new Tensor(Converts.ToByte(constant));
                            case NPTypeCode.Int16: return new Tensor(Converts.ToInt16(constant));
                            case NPTypeCode.UInt16: return new Tensor(Converts.ToUInt16(constant));
                            case NPTypeCode.Int32: return new Tensor(Converts.ToInt32(constant));
                            case NPTypeCode.UInt32: return new Tensor(Converts.ToUInt32(constant));
                            case NPTypeCode.Int64: return new Tensor(Converts.ToInt64(constant));
                            case NPTypeCode.UInt64: return new Tensor(Converts.ToUInt64(constant));
                            case NPTypeCode.Char: return new Tensor(Converts.ToByte(constant));
                            case NPTypeCode.Double: return new Tensor(Converts.ToDouble(constant));
                            case NPTypeCode.Single: return new Tensor(Converts.ToSingle(constant));
                            default:
                                throw new NotSupportedException();
                        }
                    }
            }*/
            throw new NotImplementedException("");
        }

    }
}