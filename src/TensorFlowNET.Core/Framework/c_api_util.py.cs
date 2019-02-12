using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class c_api_util
    {
        public static TF_Output tf_output(IntPtr c_op, int index) => new TF_Output(c_op, index);

        public static ImportGraphDefOptions ScopedTFImportGraphDefOptions() => new ImportGraphDefOptions();

        public static IntPtr tf_buffer(byte[] data)
        {
            if (data != null)
                throw new NotImplementedException("");
            // var buf = c_api.TF_NewBufferFromString(data);
            else
                throw new NotImplementedException("");
        }
    }
}
