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
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteBuffer(IntPtr buffer);

        /// <summary>
        /// Useful for passing *out* a protobuf.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeBufferHandle TF_NewBuffer();

        [DllImport(TensorFlowLibName)]
        public static extern TF_Buffer TF_GetBuffer(SafeBufferHandle buffer);

        /// <summary>
        /// Makes a copy of the input and sets an appropriate deallocator.  Useful for
        /// passing in read-only, input protobufs.
        /// </summary>
        /// <param name="proto">const void*</param>
        /// <param name="proto_len">size_t</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeBufferHandle TF_NewBufferFromString(IntPtr proto, ulong proto_len);
    }
}
