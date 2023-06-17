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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Tensorflow.Exceptions;
using Tensorflow.Util;
using static Tensorflow.c_api;

namespace Tensorflow
{
    /// <summary>
    /// TF_Status holds error information. It either has an OK code, or
    /// else an error code with an associated error message.
    /// </summary>
    public sealed class Status
    {
        /// <summary>
        /// Error message
        /// </summary>
        public string Message
        {
            get
            {
                using (_handle.Lease())
                {
                    return StringPiece(TF_Message(_handle));
                }
            }
        }

        /// <summary>
        /// Error code
        /// </summary>
        public TF_Code Code => TF_GetCode(_handle);

        SafeStatusHandle _handle { get; }

        public Status()
        {
            _handle = TF_NewStatus();
        }

        public Status(SafeStatusHandle handle)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        }

        public void SetStatus(TF_Code code, string msg)
        {
            TF_SetStatus(_handle, code, msg);
        }

        public bool ok() => Code == TF_Code.TF_OK;

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
                var message = Message;

                if (throwException)
                {
                    switch (Code)
                    {
                        case TF_Code.TF_OUT_OF_RANGE:
                            throw new OutOfRangeError(message);
                        case TF_Code.TF_INVALID_ARGUMENT:
                            throw new InvalidArgumentError(message);
                        default:
                            throw new NotOkStatusException(message);
                    }
                }
            }
        }

        public override string ToString()
            => $"{Code} 0x{_handle.DangerousGetHandle():x16}";

        public static implicit operator SafeStatusHandle(Status status)
        {
            return status._handle;
        }
    }
}