using System;
using System.Text;

namespace Tensorflow.NumPy
{
    internal class NumPyUtils
    {
        public static TF_DataType GetResultType(params TF_DataType[] dtypes)
        {
            var resultDType = dtypes[0];
            for(int i = 1; i < dtypes.Length; i++)
            {
                if (dtypes[i].get_datatype_size() > resultDType.get_datatype_size())
                    resultDType = dtypes[i];
            }
            return resultDType;
        }
    }
}
