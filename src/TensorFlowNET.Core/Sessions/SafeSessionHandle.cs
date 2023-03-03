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
using System.Net.NetworkInformation;
using Tensorflow.Util;

namespace Tensorflow
{
    public sealed class SafeSessionHandle : SafeTensorflowHandle
    {
        private SafeSessionHandle()
        {
        }

        public SafeSessionHandle(IntPtr handle)
            : base(handle)
        {
        }

        public override string ToString()
            => $"0x{handle:x16}";

        protected override bool ReleaseHandle()
        {
            var status = new Status();
            // c_api.TF_CloseSession(handle, tf.Status.Handle);
            c_api.TF_DeleteSession(handle, status);
            SetHandle(IntPtr.Zero);
            return true;
        }
    }
}
