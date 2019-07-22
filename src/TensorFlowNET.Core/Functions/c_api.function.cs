﻿/*****************************************************************************
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

namespace Tensorflow
{
    public partial class c_api
    {
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
        public static extern void TF_FunctionToFunctionDef(IntPtr func, IntPtr output_func_def, IntPtr status);


    }
}
