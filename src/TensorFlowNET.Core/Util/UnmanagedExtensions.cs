using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;

namespace Tensorflow.Util
{
    public static class UnmanagedExtensions
    {
        //internally UnmanagedMemoryStream can't construct with null address.
        private static readonly unsafe byte* _empty = (byte*) Marshal.AllocHGlobal(1);

        /// <summary>
        ///     Creates a memory stream based on given <paramref name="block"/>.
        /// </summary>
        /// <param name="block">The block to stream. Can be default/null.</param>
        /// <remarks>There is no need to dispose the returned <see cref="UnmanagedMemoryStream"/></remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnmanagedMemoryStream Stream(this UnmanagedMemoryBlock<byte> block)
        {
            unsafe
            {
                if (block.Address == null)
                    return new UnmanagedMemoryStream(_empty, 0);
                return new UnmanagedMemoryStream(block.Address, block.BytesCount);
            }
        }

        /// <summary>
        ///     Creates a memory stream based on given <paramref name="block"/>.
        /// </summary>
        /// <param name="block">The block to stream. Can be default/null.</param>
        /// <param name="offset">Offset from the start of the block.</param>
        /// <remarks>There is no need to dispose the returned <see cref="UnmanagedMemoryStream"/></remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnmanagedMemoryStream Stream(this UnmanagedMemoryBlock<byte> block, long offset)
        {
            if (block.BytesCount - offset <= 0)
                throw new ArgumentOutOfRangeException(nameof(offset));

            unsafe
            {
                if (block.Address == null)
                    return new UnmanagedMemoryStream(_empty, 0);
                return new UnmanagedMemoryStream(block.Address + offset, block.BytesCount - offset);
            }
        }

        /// <summary>
        ///     Creates a memory stream based on given <paramref name="address"/>.
        /// </summary>
        /// <param name="address">The block to stream. Can be IntPtr.Zero.</param>
        /// <param name="length">The length of the block in bytes.</param>
        /// <remarks>There is no need to dispose the returned <see cref="UnmanagedMemoryStream"/></remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnmanagedMemoryStream Stream(this IntPtr address, long length)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length));

            unsafe
            {
                if (address == IntPtr.Zero)
                    return new UnmanagedMemoryStream(_empty, 0);

                // ReSharper disable once AssignNullToNotNullAttribute
                return new UnmanagedMemoryStream((byte*) address, length);
            }
        }

        /// <summary>
        ///     Creates a memory stream based on given <paramref name="address"/>.
        /// </summary>
        /// <param name="address">The block to stream. Can be IntPtr.Zero.</param>
        /// <param name="offset">Offset from the start of the block.</param>
        /// <param name="length">The length of the block in bytes.</param>
        /// <remarks>There is no need to dispose the returned <see cref="UnmanagedMemoryStream"/></remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnmanagedMemoryStream Stream(this IntPtr address, long offset, long length)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length));

            unsafe
            {
                if (address == IntPtr.Zero)
                    return new UnmanagedMemoryStream(_empty, 0);

                return new UnmanagedMemoryStream((byte*) address + offset, length);
            }
        }
    }
}