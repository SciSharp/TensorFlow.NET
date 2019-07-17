using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp;
using Tensorflow;

namespace TensorFlowNet.Benchmark
{
    [SimpleJob(launchCount: 1, warmupCount: 2, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class TensorBenchmark
    {
        private double[] data;

        [GlobalSetup]
        public void Setup()
        {
            data = new double[100];
        }

        [Benchmark]
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
        public void TensorFromSpan()
        {
            var g = new Graph();
            var span = new Span<double>(data);
            for (int i = 0; i < 100; i++)
            {
                using (var tensor = new Tensor(span, new long[] { data.Length }))
                {

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
            for (int i = 0; i < 1000; i++)
            {
                using (var tensor = new Tensor(new NDArray(data)))
                {

                }
            }
        }

        //[Benchmark]
        //public void Constant()
        //{
        //    for (int i = 0; i < 100; i++)
        //    {
        //        //var tensor = new Tensor(new NDArray(data));
        //        var c = tf.constant(42.0);
        //    }
        //}

    }
}

