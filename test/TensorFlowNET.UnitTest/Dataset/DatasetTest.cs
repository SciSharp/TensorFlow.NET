using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.UnitTest.Dataset
{
    [TestClass]
    public class DatasetTest : EagerModeTestBase
    {
        [TestMethod]
        public void Range()
        {
            int iStep = 0;
            long value = 0;

            var dataset = tf.data.Dataset.range(3);
            foreach (var (step, item) in enumerate(dataset))
            {
                Assert.AreEqual(iStep, step);
                iStep++;

                Assert.AreEqual(value, (long)item.Item1);
                value++;
            }
        }

        [TestMethod]
        public void Prefetch()
        {
            int iStep = 0;
            long value = 1;

            var dataset = tf.data.Dataset.range(1, 5, 2);
            dataset = dataset.prefetch(2);

            foreach (var (step, item) in enumerate(dataset))
            {
                Assert.AreEqual(iStep, step);
                iStep++;

                Assert.AreEqual(value, (long)item.Item1);
                value += 2;
            }
        }

        [TestMethod]
        public void FromTensorSlices()
        {
            var X = tf.constant(new[] { 2013, 2014, 2015, 2016, 2017 });
            var Y = tf.constant(new[] { 12000, 14000, 15000, 16500, 17500 });

            var dataset = tf.data.Dataset.from_tensor_slices(X, Y);
            int n = 0;
            foreach (var (item_x, item_y) in dataset)
            {
                print($"x:{item_x.numpy()},y:{item_y.numpy()}");
                n += 1;
            }
            Assert.AreEqual(5, n);
        }

        [TestMethod]
        public void FromTensor()
        {
            var X = new[] { 2013, 2014, 2015, 2016, 2017 };

            var dataset = tf.data.Dataset.from_tensors(X);
            int n = 0;
            foreach (var x in dataset)
            {
                Assert.IsTrue(X.SequenceEqual(x.Item1.ToArray<int>()));
                n += 1;
            }
            Assert.AreEqual(1, n);
        }

        [TestMethod]
        public void Shard()
        {
            long value = 0;

            var dataset1 = tf.data.Dataset.range(10);
            var dataset2 = dataset1.shard(num_shards: 3, index: 0);

            foreach (var item in dataset2)
            {
                Assert.AreEqual(value, (long)item.Item1);
                value += 3;
            }

            value = 1;
            var dataset3 = dataset1.shard(num_shards: 3, index: 1);
            foreach (var item in dataset3)
            {
                Assert.AreEqual(value, (long)item.Item1);
                value += 3;
            }
        }

        [TestMethod]
        public void Skip()
        {
            long value = 7;

            var dataset = tf.data.Dataset.range(10);
            dataset = dataset.skip(7);

            foreach (var item in dataset)
            {
                Assert.AreEqual(value, (long)item.Item1);
                value++;
            }
        }

        [TestMethod]
        public void Map()
        {
            long value = 0;

            var dataset = tf.data.Dataset.range(0, 2);
            dataset = dataset.map(x => x[0] + 10);

            foreach (var item in dataset)
            {
                Assert.AreEqual(value + 10, (long)item.Item1);
                value++;
            }
        }

        [TestMethod]
        public void Cache()
        {
            long value = 0;

            var dataset = tf.data.Dataset.range(5);
            dataset = dataset.cache();

            foreach (var item in dataset)
            {
                Assert.AreEqual(value, (long)item.Item1);
                value++;
            }
        }

        [TestMethod]
        public void Cardinality()
        {
            var dataset = tf.data.Dataset.range(10);
            var cardinality = dataset.cardinality();
            Assert.AreEqual(cardinality.numpy(), 10L);
            dataset = dataset.map(x => x[0] + 1);
            cardinality = dataset.cardinality();
            Assert.AreEqual(cardinality.numpy(), 10L);
        }

        [TestMethod]
        public void CardinalityWithAutoTune()
        {
            var dataset = tf.data.Dataset.range(10);
            dataset = dataset.map(x => x, num_parallel_calls: -1);
            var cardinality = dataset.cardinality();
            Assert.AreEqual(cardinality.numpy(), 10L);
        }

        [TestMethod]
        public void CardinalityWithRepeat()
        {
            var dataset = tf.data.Dataset.range(10);
            dataset = dataset.repeat();
            var cardinality = dataset.cardinality();
            Assert.IsTrue((cardinality == tf.data.INFINITE_CARDINALITY).numpy());

            dataset = dataset.filter(x => true);
            cardinality = dataset.cardinality();
            Assert.IsTrue((cardinality == tf.data.UNKNOWN_CARDINALITY).numpy());
        }

        [TestMethod]
        public void Shuffle()
        {
            tf.set_random_seed(1234);

            var dataset = tf.data.Dataset.range(3);
            var shuffled = dataset.shuffle(3);

            var zipped = tf.data.Dataset.zip(dataset, shuffled);

            bool allEqual = true;
            foreach (var item in zipped)
            {
                if (item.Item1 != item.Item2)
                    allEqual = false;
            }

            Assert.IsFalse(allEqual);
        }
        [Ignore]
        [TestMethod]
        public void GetData()
        {
            var vocab_size = 20000; // Only consider the top 20k words
            var maxlen = 200; // Only consider the first 200 words of each movie review
            var dataset = keras.datasets.imdb.load_data(num_words: vocab_size, maxlen: maxlen);
            var x_train = dataset.Train.Item1;
            var y_train = dataset.Train.Item2;
            var x_val = dataset.Test.Item1;
            var y_val = dataset.Test.Item2;

            x_train = keras.preprocessing.sequence.pad_sequences(RemoveZeros(x_train), maxlen: maxlen);
            x_val = keras.preprocessing.sequence.pad_sequences(RemoveZeros(x_val), maxlen: maxlen);
            print(len(x_train) + " Training sequences");
            print(len(x_val) + " Validation sequences");
        }
        IEnumerable<int[]> RemoveZeros(NDArray data)
        {
            var data_array = (int[,])data.ToMultiDimArray<int>();
            List<int[]> new_data = new List<int[]>();
            for (var i = 0; i < data_array.GetLength(0); i++)
            {
                List<int> new_array = new List<int>();
                for (var j = 0; j < data_array.GetLength(1); j++)
                {
                    if (data_array[i, j] == 0)
                        break;
                    else
                        new_array.Add(data_array[i, j]);
                }
                new_data.Add(new_array.ToArray());
            }
            return new_data;
        }
    }
}
