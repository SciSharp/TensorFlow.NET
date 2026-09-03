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

        /// <summary>
        /// Record &lt;key, value&gt; as a payload in *s. Overwrites previous payload with same key.
        /// No-op when status is TF_OK.
        /// </summary>
        /// <param name="s">TF_Status*</param>
        /// <param name="key">null-terminated key string</param>
        /// <param name="value">null-terminated value string</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetPayload(SafeStatusHandle s, string key, string value);

        /// <summary>
        /// Iterates over stored payloads and calls visitor(key, value, capture) for each one.
        /// key and value are only valid during the callback.
        /// </summary>
        /// <param name="s">const TF_Status*</param>
        /// <param name="visitor">callback invoked per payload entry</param>
        /// <param name="capture">passed to visitor unmodified</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TF_PayloadVisitor(IntPtr key, IntPtr value, IntPtr capture);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_ForEachPayload(SafeStatusHandle s, TF_PayloadVisitor visitor, IntPtr capture);

        /// <summary>
        /// Converts from an I/O error code (e.g., errno) to a TF_Status value.
        /// Any previous information is lost. Prefer over TF_SetStatus for I/O errors.
        /// </summary>
        /// <param name="s">TF_Status*</param>
        /// <param name="error_code">errno-style error code</param>
        /// <param name="context">null-terminated context string</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetStatusFromIOError(SafeStatusHandle s, int error_code, string context);
    }
}
