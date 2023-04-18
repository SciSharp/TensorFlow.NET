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
using System.IO;
using System.Runtime.CompilerServices;
using Tensorflow.Util;
using static Tensorflow.c_api;

namespace Tensorflow
{
    /// <summary>
    ///     Represents a TF_Buffer that can be passed to Tensorflow.
    /// </summary>
    public sealed class Buffer
    {
        SafeBufferHandle _handle;

        /// <remarks>
        /// <inheritdoc cref="SafeHandleLease" path="/devdoc/usage"/>
        /// </remarks>
        private unsafe ref readonly TF_Buffer DangerousBuffer
            => ref Unsafe.AsRef<TF_Buffer>(_handle.DangerousGetHandle().ToPointer());

        /// <summary>
        ///     The memory block representing this buffer.
        /// </summary>
        /// <remarks>
        /// <para>The deallocator is set to null.</para>
        ///
        /// <inheritdoc cref="SafeHandleLease" path="/devdoc/usage"/>
        /// </remarks>
        public unsafe MemoryStream DangerousMemoryBlock
        {
            get
            {
                ref readonly TF_Buffer buffer = ref DangerousBuffer;
                return new MemoryStream(ToArray());
            }
        }

        /// <summary>
        ///     The bytes length of this buffer.
        /// </summary>
        public ulong Length
        {
            get
            {
                using (_handle.Lease())
                {
                    return DangerousBuffer.length;
                }
            }
        }

        public Buffer()
            => _handle = TF_NewBuffer();

        public Buffer(SafeBufferHandle handle)
            => _handle = handle;

        public Buffer(byte[] data)
            => _handle = _toBuffer(data);

        private static SafeBufferHandle _toBuffer(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            unsafe
            {
                fixed (byte* src = data)
                    return TF_NewBufferFromString(new IntPtr(src), (ulong)data.LongLength);
            }
        }

        /// <summary>
        ///     Copies this buffer's contents onto a <see cref="byte"/> array.
        /// </summary>
        public unsafe byte[] ToArray()
        {
            using (_handle.Lease())
            {
                ref readonly TF_Buffer buffer = ref DangerousBuffer;

                if (buffer.length == 0)
                    return new byte[0];

                var data = new byte[DangerousBuffer.length];
                fixed (byte* dst = data)
                    System.Buffer.MemoryCopy(buffer.data.ToPointer(), dst, buffer.length, buffer.length);

                return data;
            }
        }

        public void Release()
        {
            _handle.Dispose();
            _handle = null;
        }

        public override string ToString()
            => $"0x{_handle.DangerousGetHandle():x16}";

        public static implicit operator SafeBufferHandle(Buffer buffer)
        {
            return buffer._handle;
        }
    }
}