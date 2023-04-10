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
using System.Runtime.InteropServices;
using Tensorflow.Functions;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteFunction(IntPtr handle);

        /// <summary>
        /// Write out a serialized representation of `func` (as a FunctionDef protocol
        /// message) to `output_func_def` (allocated by TF_NewBuffer()).
        /// `output_func_def`'s underlying buffer will be freed when TF_DeleteBuffer()
        /// is called.
        /// </summary>
        /// <param name="func"></param>
        /// <param name="output_func_def"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_FunctionToFunctionDef(SafeFuncGraphHandle func, SafeBufferHandle output_func_def, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern SafeFuncGraphHandle TF_GraphToFunction(SafeGraphHandle fn_body, string fn_name,
            bool append_hash_to_fn_name,
            int num_opers, IntPtr[] opers,
            int ninputs, TF_Output[] inputs,
            int noutputs, TF_Output[] outputs,
            string[] output_names,
            IntPtr opts,
            string description,
            SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_FunctionSetAttrValueProto(SafeFuncGraphHandle func, string attr_name, byte[] proto, int proto_len, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_FunctionName(SafeFuncGraphHandle func);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphCopyFunction(SafeGraphHandle g, SafeFuncGraphHandle func, SafeFuncGraphHandle grad, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_GraphGetFunctions(SafeGraphHandle g, IntPtr[] funcs, int max_func, SafeStatusHandle status);
    }
}
