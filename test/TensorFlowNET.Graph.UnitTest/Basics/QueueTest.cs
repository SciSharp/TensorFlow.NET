using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class QueueTest : GraphModeTestBase
    {
        [TestMethod]
        public void PaddingFIFOQueue()
        {
            var numbers = tf.placeholder(tf.int32);
            var queue = tf.PaddingFIFOQueue(10, tf.int32, new Shape(-1));
            var enqueue = queue.enqueue(numbers);
            var dequeue_many = queue.dequeue_many(n: 3);

            var sess = tf.Session();
            sess.run(enqueue, (numbers, new[] { 1 }));
            sess.run(enqueue, (numbers, new[] { 2, 3 }));
            sess.run(enqueue, (numbers, new[] { 3, 4, 5 }));

            var result = sess.run(dequeue_many[0]);

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0 }, result[0].ToArray<int>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 2, 3, 0 }, result[1].ToArray<int>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 4, 5 }, result[2].ToArray<int>()));
        }

        [TestMethod]
        public void FIFOQueue()
        {
            // create a first in first out queue with capacity up to 2
            // and data type set as int32
            var queue = tf.FIFOQueue(2, tf.int32);
            // init queue, push 3 elements into queue.
            var init = queue.enqueue_many(new[] { 10, 20 });
            // pop out the first element
            var x = queue.dequeue();
            // add 1
            var y = x + 1;
            // push back into queue
            var inc = queue.enqueue(y);

            var sess = tf.Session();
            // init queue
            init.run();

            // pop out first element and push back calculated y
            (int dequeued, _) = sess.run((x, inc));
            Assert.AreEqual(10, dequeued);

            (dequeued, _) = sess.run((x, inc));
            Assert.AreEqual(20, dequeued);

            (dequeued, _) = sess.run((x, inc));
            Assert.AreEqual(11, dequeued);

            (dequeued, _) = sess.run((x, inc));
            Assert.AreEqual(21, dequeued);

            // thread will hang or block if you run sess.run(x) again
            // until queue has more element.
        }

        [TestMethod]
        public void PriorityQueue()
        {
            var queue = tf.PriorityQueue(3, tf.@string);
            var init = queue.enqueue_many(new[] { 2L, 4L, 3L }, new[] { "p1", "p2", "p3" });
            var x = queue.dequeue();

            var sess = tf.Session();
            init.run();

            var result = sess.run(x);
            Assert.AreEqual(result[0], 2L);

            result = sess.run(x);
            Assert.AreEqual(result[0], 3L);

            result = sess.run(x);
            Assert.AreEqual(result[0], 4L);
        }

        [TestMethod]
        public void RandomShuffleQueue()
        {
            var queue = tf.RandomShuffleQueue(10, min_after_dequeue: 1, dtype: tf.int32);
            var init = queue.enqueue_many(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var x = queue.dequeue();

            string results = "";
            var sess = tf.Session();
            init.run();

            foreach (var i in range(9))
                results += (int)sess.run(x) + ".";

            // output in random order
            Assert.IsFalse(results == "1.2.3.4.5.6.7.8.9.");
        }
    }
}
