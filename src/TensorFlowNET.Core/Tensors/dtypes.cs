using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static class dtypes
    {
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
                case TF_DataType.TF_INT32:
                    dtype = DataType.DtInt32;
                    break;
                case TF_DataType.TF_FLOAT:
                    dtype = DataType.DtFloat;
                    break;
                case TF_DataType.TF_DOUBLE:
                    dtype = DataType.DtDouble;
                    break;
                case TF_DataType.TF_STRING:
                    dtype = DataType.DtString;
                    break;
                default:
                    throw new Exception("Not Implemented");
            }

            return dtype;
        }
    }
}
