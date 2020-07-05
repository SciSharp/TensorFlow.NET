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

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Delete a previously created status object.
        /// </summary>
        /// <param name="s"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteStatus(IntPtr s);

        /// <summary>
        /// Return the code record in *s.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_Code TF_GetCode(SafeStatusHandle s);

        /// <summary>
        /// Return a pointer to the (null-terminated) error message in *s.
        /// The return value points to memory that is only usable until the next
        /// mutation to *s.  Always returns an empty string if TF_GetCode(s) is TF_OK.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_Message(SafeStatusHandle s);

        /// <summary>
        /// Return a new status object.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeStatusHandle TF_NewStatus();

        /// <summary>
        /// Record &lt;code, msg> in *s.  Any previous information is lost.
        /// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
        /// </summary>
        /// <param name="s"></param>
        /// <param name="code"></param>
        /// <param name="msg"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetStatus(SafeStatusHandle s, TF_Code code, string msg);
    }
}
