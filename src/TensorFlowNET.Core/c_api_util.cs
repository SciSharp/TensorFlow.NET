using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class c_api_util
    {
        public static TF_Output tf_output(IntPtr c_op, int index)
        {
            
            var ret = new TF_Output();
            ret.oper = c_op;
            ret.index = index;

            return ret;
        }
    }
}
