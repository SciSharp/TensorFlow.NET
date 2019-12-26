using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public class TF_ImportGraphDefResults : DisposableObject
    {
        /*public IntPtr return_nodes;
        public IntPtr missing_unused_key_names;
        public IntPtr missing_unused_key_indexes;
        public IntPtr missing_unused_key_names_data;*/

        public TF_ImportGraphDefResults(IntPtr handle)
        {
            _handle = handle;
        }

        public TF_Output[] return_tensors
        {
            get
            {
                IntPtr return_output_handle = IntPtr.Zero;
                int num_outputs = -1;
                c_api.TF_ImportGraphDefResultsReturnOutputs(_handle, ref num_outputs, ref return_output_handle);
                TF_Output[] return_outputs = new TF_Output[num_outputs];
                unsafe
                {
                    var tf_output_ptr = (TF_Output*)return_output_handle;
                    for (int i = 0; i < num_outputs; i++)
                        return_outputs[i] = *(tf_output_ptr + i);
                    return return_outputs;
                }
            }
        }

        public TF_Operation[] return_opers
        {
            get
            {
                return new TF_Operation[0];
                /*TF_Operation return_output_handle = new TF_Operation();
                int num_outputs = -1;
                c_api.TF_ImportGraphDefResultsReturnOperations(_handle, ref num_outputs, ref return_output_handle);
                TF_Operation[] return_outputs = new TF_Operation[num_outputs];
                unsafe
                {
                    var tf_output_ptr = (TF_Operation*)return_output_handle;
                    for (int i = 0; i < num_outputs; i++)
                        return_outputs[i] = *(tf_output_ptr + i);
                    return return_outputs;
                }*/
            }
        }

        public static implicit operator TF_ImportGraphDefResults(IntPtr handle)
            => new TF_ImportGraphDefResults(handle);

        public static implicit operator IntPtr(TF_ImportGraphDefResults results)
            => results._handle;

        protected override void DisposeUnmanagedResources(IntPtr handle)
            => c_api.TF_DeleteImportGraphDefResults(handle);
    }
}
