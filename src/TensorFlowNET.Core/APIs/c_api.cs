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
    /// <summary>
    /// C API for TensorFlow.
    /// Port from tensorflow\c\c_api.h
    /// 
    /// The API leans towards simplicity and uniformity instead of convenience
    /// since most usage will be by language specific wrappers.
    /// 
    /// The params type mapping between c_api and .NET
    /// TF_XX** => ref IntPtr (TF_Operation** op) => (ref IntPtr op)
    /// TF_XX* => IntPtr (TF_Graph* graph) => (IntPtr graph)
    /// struct => struct (TF_Output output) => (TF_Output output)
    /// struct* => struct[] (TF_Output* output) => (TF_Output[] output)
    /// struct* => struct* for ref
    /// const char* => string
    /// int32_t => int
    /// int64_t* => long[]
    /// size_t* => ulong[]
    /// size_t* => ref ulong
    /// void* => IntPtr
    /// string => IntPtr c_api.StringPiece(IntPtr)
    /// unsigned char => byte
    /// </summary>
    public partial class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        public static string StringPiece(IntPtr handle)
        {
            return handle == IntPtr.Zero ? String.Empty : Marshal.PtrToStringAnsi(handle);
        }

        public delegate void Deallocator(IntPtr data, IntPtr size, ref DeallocatorArgs args);

        public struct DeallocatorArgs
        {
            public bool deallocator_called;
            public IntPtr gc_handle;
        }

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
