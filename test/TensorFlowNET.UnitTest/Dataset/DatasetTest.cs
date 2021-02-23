using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using static Tensorflow.Binding;

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

            var dataset = tf.data.Dataset.from_tensor(X);
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
            dataset = dataset.map(x => x[0] + 1);
            var cardinality = dataset.dataset_cardinality();
            Assert.AreEqual(new long[] { 10 }, cardinality.numpy());
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
    }
}
