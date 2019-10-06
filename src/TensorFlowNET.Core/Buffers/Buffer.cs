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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;
using static Tensorflow.c_api;

namespace Tensorflow
{
    /// <summary>
    ///     Represents a TF_Buffer that can be passed to Tensorflow.
    /// </summary>
    public class Buffer : DisposableObject
    {
        private unsafe TF_Buffer buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => *bufferptr;
        }

        private unsafe TF_Buffer* bufferptr
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (TF_Buffer*) _handle;
        }

        /// <summary>
        ///     The memory block representing this buffer.
        /// </summary>
        /// <remarks>The deallocator is set to null.</remarks>
        public UnmanagedMemoryBlock<byte> MemoryBlock
        {
            get
            {
                unsafe
                {
                    EnsureNotDisposed();
                    var buff = (TF_Buffer*) _handle;
                    return new UnmanagedMemoryBlock<byte>((byte*) buff->data.ToPointer(), (long) buff->length);
                }
            }
        }

        /// <summary>
        ///     The bytes length of this buffer.
        /// </summary>
        public ulong Length
        {
            get
            {
                EnsureNotDisposed();
                return buffer.length;
            }
        }

        public Buffer() => _handle = TF_NewBuffer();

        public Buffer(IntPtr handle)
        {
            if (handle == IntPtr.Zero)
                throw new ArgumentException("Handle (IntPtr) can't be zero.", nameof(handle));

            _handle = handle;
        }

        public Buffer(byte[] data) : this(_toBuffer(data))
        { }

        private static IntPtr _toBuffer(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            unsafe
            {
                fixed (byte* src = data)
                    return TF_NewBufferFromString(new IntPtr(src), (ulong) data.LongLength);
            }
        }

        public static implicit operator IntPtr(Buffer buffer)
        {
            buffer.EnsureNotDisposed();
            return buffer._handle;
        }

        public static explicit operator byte[](Buffer buffer) => buffer.ToArray(); //has to be explicit, developer will assume it doesn't cost.

        /// <summary>
        ///     Copies this buffer's contents onto a <see cref="byte"/> array.
        /// </summary>
        public byte[] ToArray()
        {
            EnsureNotDisposed();

            unsafe
            {
                var len = buffer.length;
                if (len == 0)
                    return Array.Empty<byte>();

                byte[] data = new byte[len];
                fixed (byte* dst = data)
                    System.Buffer.MemoryCopy((void*) bufferptr->data, dst, len, len);

                return data;
            }
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            TF_DeleteBuffer(handle);
        }
    }
}