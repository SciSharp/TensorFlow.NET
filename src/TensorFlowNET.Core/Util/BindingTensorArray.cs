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
    public class BindingTensorArray : DisposableObject
    {
        TF_BindingArray data;
        public IntPtr Address => data.array;
        public int Length => data.length;

        public BindingTensorArray(IntPtr handle) : base(handle)
        {
            if (_handle != IntPtr.Zero)
                data = Marshal.PtrToStructure<TF_BindingArray>(_handle);
            else
                data = default;
        }

        public static implicit operator BindingTensorArray(IntPtr handle)
            => new BindingTensorArray(handle);

        public unsafe IntPtr this[int index]
            => data[index];

        public unsafe IntPtr[] Data
            => data.Data;

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            c_api.TFE_DeleteBindingTensorArray(_handle);
        }
    }
}
