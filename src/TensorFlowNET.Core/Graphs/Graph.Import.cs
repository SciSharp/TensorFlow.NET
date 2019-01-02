using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Graph
    {
        public unsafe TF_Output[] ImportGraphDefWithReturnOutputs(Buffer graph_def, ImportGraphDefOptions opts, Status s)
        {
            var num_return_outputs = opts.NumReturnOutputs;
            var return_outputs = new TF_Output[num_return_outputs];
            int size = Marshal.SizeOf<TF_Output>();
            var return_output_handle = Marshal.AllocHGlobal(size * num_return_outputs);

            c_api.TF_GraphImportGraphDefWithReturnOutputs(_handle, graph_def, opts, return_output_handle, num_return_outputs, s);
            for (int i = 0; i < num_return_outputs; i++)
            {
                var handle = return_output_handle + i * size;
                return_outputs[i] = Marshal.PtrToStructure<TF_Output>(handle);
            }

            return return_outputs;
        }
    }
}
