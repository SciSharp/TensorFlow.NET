using System;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public unsafe void memcpy<T>(T* dst, void* src, ulong size)
            where T : unmanaged
        {
            System.Buffer.MemoryCopy(src, dst, size, size);
        }

        public unsafe void memcpy<T>(void* dst, T* src, ulong size)
            where T : unmanaged
        {
            System.Buffer.MemoryCopy(src, dst, size, size);
        }

        public unsafe void memcpy(void* dst, IntPtr src, ulong size)
        {
            System.Buffer.MemoryCopy(src.ToPointer(), dst, size, size);
        }

        public unsafe void memcpy<T>(T[] dst, IntPtr src, ulong size)
            where T : unmanaged
        {
            fixed (void* p = &dst[0])
                System.Buffer.MemoryCopy(src.ToPointer(), p, size, size);
        }

        public unsafe void memcpy<T>(T[] dst, IntPtr src, long size)
            where T : unmanaged
        {
            fixed (void* p = &dst[0])
                System.Buffer.MemoryCopy(src.ToPointer(), p, size, size);
        }

        public unsafe void memcpy<T>(IntPtr dst, T[] src, ulong size)
            where T : unmanaged
        {
            if (src.Length == 0) return;

            fixed (void* p = &src[0])
                System.Buffer.MemoryCopy(p, dst.ToPointer(), size, size);
        }

        public unsafe void memcpy<T>(IntPtr dst, T[] src, long size)
            where T : unmanaged
        {
            if (src.Length == 0) return;

            fixed (void* p = &src[0])
                System.Buffer.MemoryCopy(p, dst.ToPointer(), size, size);
        }
    }
}
