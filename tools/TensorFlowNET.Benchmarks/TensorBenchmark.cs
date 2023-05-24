using BenchmarkDotNet.Attributes;

namespace TensorFlowBenchmark
{
    [SimpleJob(launchCount: 1, warmupCount: 1)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class TensorBenchmark
    {
        private double[] data;

        [GlobalSetup]
        public void Setup()
        {
            data = new double[100];
        }

        /*[Benchmark]
        public void ScalarTensor()
        {
            var g = new Graph();
            for (int i = 0; i < 100; i++)
            {
                using (var tensor = new Tensor(17.0))
                {

                }
            }
        }

        [Benchmark]
        public unsafe void TensorFromFixedPtr()
        {
            var g = new Graph();
            for (int i = 0; i < 100; i++)
            {
                fixed (double* ptr = &data[0])
                {
                    using (var t = new Tensor((IntPtr)ptr, new long[] { data.Length }, tf.float64, 8 * data.Length))
                    {
                    }
                }
            }
        }

        [Benchmark]
        public void TensorFromArray()
        {
            var g=new Graph();
            for (int i = 0; i < 100; i++)
            {
                using (var tensor = new Tensor(data))
                {

                }
            }
        }


        [Benchmark]
        public void TensorFromNDArray()
        {
            var g = new Graph();
            for (int i = 0; i < 100; i++)
            {
                using (var tensor = new Tensor(new NDArray(data)))
                {

                }
            }
        }*/
    }
}

