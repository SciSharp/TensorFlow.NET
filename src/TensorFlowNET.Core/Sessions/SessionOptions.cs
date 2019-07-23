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
    public class SessionOptions : IDisposable
    {
        private IntPtr _handle;
        private Status _status;

        public unsafe SessionOptions()
        {
            var opts = c_api.TF_NewSessionOptions();
            _handle = opts;
            _status = new Status();
        }

        public unsafe SessionOptions(IntPtr handle)
        {
            _handle = handle;
        }

        public void Dispose()
        {
            c_api.TF_DeleteSessionOptions(_handle);
            _status.Dispose();
        }

        public Status SetConfig(ConfigProto config)
        {
            var bytes = config.ToByteArray();
            var proto = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, proto, bytes.Length);
            c_api.TF_SetConfig(_handle, proto, (ulong)bytes.Length, _status);
            _status.Check(false);
            return _status;
        }

        public static implicit operator IntPtr(SessionOptions opts) => opts._handle;
        public static implicit operator SessionOptions(IntPtr handle) => new SessionOptions(handle);
    }
}
