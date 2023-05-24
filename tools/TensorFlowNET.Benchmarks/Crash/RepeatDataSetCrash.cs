using BenchmarkDotNet.Attributes;
using System;
using System.Collections.Generic;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Benchmark.Crash
{
    public class RepeatDataSetCrash
    {
        [Benchmark]
        public void Run()
        {
            var data = tf.convert_to_tensor(np.arange(0, 50000 * 10).astype(np.float32).reshape((50000, 10)));

            var dataset = keras.preprocessing.timeseries_dataset_from_array(data,
                sequence_length: 10,
                sequence_stride: 1,
                shuffle: false,
                batch_size: 32);

            while (true)
                foreach (var d in dataset)
                    ;
        }
    }
}
