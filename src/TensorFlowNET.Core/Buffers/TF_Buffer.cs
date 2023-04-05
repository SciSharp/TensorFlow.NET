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
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Buffer
    {
        public IntPtr data;
        public ulong length;
        public IntPtr data_deallocator;

        public unsafe Span<T> AsSpan<T>() where T: unmanaged
        {
            if(length > int.MaxValue)
            {
                throw new ValueError($"The length {length} is too large to use in the span.");
            }
            return new Span<T>(data.ToPointer(), (int)length);
        }

        public unsafe byte[] ToByteArray()
        {
            byte[] res = new byte[length];
            if(length > int.MaxValue)
            {
                byte* root = (byte*)data;
                for(ulong i = 0; i < length; i++)
                {
                    res[i] = *(root++);
                }
            }
            else
            {
                new Span<byte>(data.ToPointer(), (int)length).CopyTo(res.AsSpan());
            }
            return res;
        }
    }
}
