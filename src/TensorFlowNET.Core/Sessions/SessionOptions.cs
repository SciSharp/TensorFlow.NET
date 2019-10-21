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

using Google.Protobuf;
using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    internal class SessionOptions : DisposableObject
    {
        public SessionOptions(string target = "", ConfigProto config = null)
        {
            _handle = c_api.TF_NewSessionOptions();
            c_api.TF_SetTarget(_handle, target);
            if (config != null)
                SetConfig(config);
        }

        public SessionOptions(IntPtr handle)
        {
            _handle = handle;
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
            => c_api.TF_DeleteSessionOptions(handle);

        private void SetConfig(ConfigProto config)
        {
            var bytes = config.ToByteArray();
            var proto = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, proto, bytes.Length);

            using (var status = new Status())
            {
                c_api.TF_SetConfig(_handle, proto, (ulong)bytes.Length, status);
                status.Check(false);
            }

            Marshal.FreeHGlobal(proto);
        }

        public static implicit operator IntPtr(SessionOptions opts) => opts._handle;
        public static implicit operator SessionOptions(IntPtr handle) => new SessionOptions(handle);
    }
}
