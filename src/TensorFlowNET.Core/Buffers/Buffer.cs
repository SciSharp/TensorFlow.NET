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
    public class Buffer : DisposableObject
    {
        private TF_Buffer buffer => Marshal.PtrToStructure<TF_Buffer>(_handle);

        public byte[] Data 
        {
            get 
            {
                var data = new byte[buffer.length];
                if (data.Length > 0)
                    Marshal.Copy(buffer.data, data, 0, data.Length);
                return data;
            }
        }

        public int Length => (int)buffer.length;

        public Buffer()
        {
            _handle = c_api.TF_NewBuffer();
        }

        public Buffer(IntPtr handle)
        {
            _handle = handle;
        }

        public Buffer(byte[] data)
        {
            var dst = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dst, data.Length);

            _handle = c_api.TF_NewBufferFromString(dst, (ulong)data.Length);

            Marshal.FreeHGlobal(dst);
        }

        public static implicit operator IntPtr(Buffer buffer)
        {
            return buffer._handle;
        }

        public static implicit operator byte[](Buffer buffer)
        {
            return buffer.Data;
        }

        protected override void DisposeUnManagedState(IntPtr handle)
            => c_api.TF_DeleteBuffer(handle);
    }
}
