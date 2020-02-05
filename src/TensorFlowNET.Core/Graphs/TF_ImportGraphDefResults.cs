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

using System;

namespace Tensorflow
{
    public sealed class TF_ImportGraphDefResults : IDisposable
    {
        /*public IntPtr return_nodes;
        public IntPtr missing_unused_key_names;
        public IntPtr missing_unused_key_indexes;
        public IntPtr missing_unused_key_names_data;*/

        private SafeImportGraphDefResultsHandle Handle { get; }

        public TF_ImportGraphDefResults(SafeImportGraphDefResultsHandle handle)
        {
            Handle = handle;
        }

        public TF_Output[] return_tensors
        {
            get
            {
                IntPtr return_output_handle = IntPtr.Zero;
                int num_outputs = -1;
                c_api.TF_ImportGraphDefResultsReturnOutputs(Handle, ref num_outputs, ref return_output_handle);
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

        public void Dispose()
            => Handle.Dispose();
    }
}
