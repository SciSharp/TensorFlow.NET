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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using static Tensorflow.c_api;

namespace Tensorflow
{
    /// <summary>
    /// TF_Status holds error information. It either has an OK code, or
    /// else an error code with an associated error message.
    /// </summary>
    public class Status : DisposableObject
    {
        /// <summary>
        /// Error message
        /// </summary>
        public string Message => c_api.StringPiece(TF_Message(_handle));

        /// <summary>
        /// Error code
        /// </summary>
        public TF_Code Code => TF_GetCode(_handle);

        public Status()
        {
            _handle = TF_NewStatus();
        }

        public void SetStatus(TF_Code code, string msg)
        {
            TF_SetStatus(_handle, code, msg);
        }

        /// <summary>
        /// Check status 
        /// Throw exception with error message if code != TF_OK
        /// </summary>
        /// <exception cref="TensorflowException">When the returned check is not TF_Code.TF_OK</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DebuggerHidden]
        public void Check(bool throwException = false)
        {
            if (Code != TF_Code.TF_OK)
            {
                Console.WriteLine(Message);
                if (throwException)
                    throw new TensorflowException(Message);
            }
        }

        public static implicit operator IntPtr(Status status)
            => status._handle;

        protected override void DisposeUnmanagedResources(IntPtr handle)
            => TF_DeleteStatus(handle);
    }
}