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
    public class TensorCreation
    {
        private double[] data;

        [GlobalSetup]
        public void Setup()
        {
            data = new double[10];
        }

        [Benchmark]
        public void TensorFromArray()
        {
            var g=new Graph();
            for (int i = 0; i < 1000; i++)
            {
                var tensor = new Tensor(data);
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
                var tensor = new Tensor(new NDArray(data));
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

