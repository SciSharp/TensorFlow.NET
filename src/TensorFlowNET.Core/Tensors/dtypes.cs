using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static class dtypes
    {
        public static Type as_numpy_datatype(this TF_DataType type)
        {
            switch (type)
            {
                case TF_DataType.TF_INT32:
                    return typeof(int);
                case TF_DataType.TF_INT16:
                    return typeof(short);
                case TF_DataType.TF_FLOAT:
                    return typeof(float);
                case TF_DataType.TF_DOUBLE:
                    return typeof(double);
                case TF_DataType.TF_STRING:
                    return typeof(string);
                default:
                    throw new NotImplementedException("as_numpy_datatype failed");
            }
        }

        public static TF_DataType as_dtype(Type type)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;

            switch (type.Name)
            {
                case "Int32":
                    dtype = TF_DataType.TF_INT32;
                    break;
                case "Single":
                    dtype = TF_DataType.TF_FLOAT;
                    break;
                case "Double":
                    dtype = TF_DataType.TF_DOUBLE;
                    break;
                case "String":
                    dtype = TF_DataType.TF_STRING;
                    break;
                default:
                    throw new Exception("Not Implemented");
            }

            return dtype;
        }

        public static DataType as_datatype_enum(this TF_DataType type)
        {
            DataType dtype = DataType.DtInvalid;

            switch (type)
            {
                default:
                    Enum.TryParse(((int)type).ToString(), out dtype);
                    break;
            }

            return dtype;
        }

        public static TF_DataType as_base_dtype(this TF_DataType type)
        {
            return (int)type > 100 ?
                (TF_DataType)Enum.Parse(typeof(TF_DataType), ((int)type - 100).ToString()) :
                type;
        }

        public static DataType as_base_dtype(this DataType type)
        {
            return (int)type > 100 ?
                (DataType)Enum.Parse(typeof(DataType), ((int)type - 100).ToString()) :
                type;
        }

        public static bool is_complex(this TF_DataType type)
        {
            return type == TF_DataType.TF_COMPLEX || type == TF_DataType.TF_COMPLEX64 || type == TF_DataType.TF_COMPLEX128;
        }

        public static bool is_integer(this TF_DataType type)
        {
            return type == TF_DataType.TF_INT8 || type == TF_DataType.TF_INT16 || type == TF_DataType.TF_INT32 || type == TF_DataType.TF_INT64;
        }

        public static bool is_floating(this TF_DataType type)
        {
            return type == TF_DataType.TF_HALF || type == TF_DataType.TF_FLOAT || type == TF_DataType.TF_DOUBLE;
        }
    }
}
