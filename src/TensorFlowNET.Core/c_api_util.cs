using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class c_api_util
    {
        public static TF_Output tf_output(IntPtr c_op, int index)
        {
            return new TF_Output(c_op, index);
        }
    }
}
