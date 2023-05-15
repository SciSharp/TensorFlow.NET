using BenchmarkDotNet.Attributes;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorFlowBenchmark.Unmanaged
{
    public struct UnmanagedStruct
    {
        public int a;
        public long b;
        public UnmanagedStruct(int _)
        {
            a = 2;
            b = 3;
        }
    }

    [SimpleJob(launchCount: 1, warmupCount: 2)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public unsafe class StructCastBenchmark
    {
        private static void EnsureIsUnmanaged<T>(T _) where T : unmanaged
        { }

        static StructCastBenchmark() //if UnmanagedStruct is not unmanaged struct then this will fail to compile.
            => EnsureIsUnmanaged(new UnmanagedStruct());

        private IntPtr data;
        private void* dataptr;

        [GlobalSetup]
        public void Setup()
        {
            data = Marshal.AllocHGlobal(Marshal.SizeOf<UnmanagedStruct>());
            dataptr = data.ToPointer();
        }

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void Marshal_PtrToStructure()
        {
            UnmanagedStruct _;
            for (int i = 0; i < 10000; i++)
            {
                _ = Marshal.PtrToStructure<UnmanagedStruct>(data);
            }
        }

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void PointerCast()
        {
            var dptr = dataptr;
            UnmanagedStruct _;
            for (int i = 0; i < 10000; i++)
            {
                _ = *(UnmanagedStruct*)dptr;
            }
        }

        [Benchmark, MethodImpl(MethodImplOptions.NoOptimization)]
        public void Unsafe_Read()
        {
            var dptr = dataptr;
            UnmanagedStruct _;
            for (int i = 0; i < 10000; i++)
            {
                _ = Unsafe.Read<UnmanagedStruct>(dptr);
            }
        }

    }
}