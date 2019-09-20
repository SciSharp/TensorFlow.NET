using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class QueueTest
    {
        [TestMethod]
        public void PaddingFIFOQueue()
        {
            var numbers = tf.placeholder(tf.int32);
            var queue = tf.PaddingFIFOQueue(capacity: 10, dtypes: new[] { tf.int32 }, shapes: new[] { new TensorShape(-1) });
            var enqueue = queue.enqueue(numbers);
            var dequeue_many = queue.dequeue_many(n: 3);

            using(var sess = tf.Session())
            {
                sess.run(enqueue, (numbers, new[] { 1 }));
                sess.run(enqueue, (numbers, new[] { 2, 3 }));
                sess.run(enqueue, (numbers, new[] { 3, 4, 5 }));

                var result = sess.run(dequeue_many[0]);

                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0 }, result[0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 2, 3, 0 }, result[1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 4, 5 }, result[2].ToArray<int>()));
            }
        }
    }
}
