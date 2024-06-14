using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Training
{
    [TestClass]
    public class GradientDescentOptimizerTest : PythonTest
    {
        private static TF_DataType GetTypeForNumericType<T>() where T : struct
        {
            return Type.GetTypeCode(typeof(T)) switch
            {
                TypeCode.Single => np.float32,
                TypeCode.Double => np.float64,
                _ => throw new NotImplementedException(),
            };
        }

        private void TestBasic<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            // train.GradientDescentOptimizer is V1 only API.
            tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var var0 = tf.Variable(new[] { 1.0, 2.0 }, dtype: dtype);
                var var1 = tf.Variable(new[] { 3.0, 4.0 }, dtype: dtype);
                var grads0 = tf.constant(new[] { 0.1, 0.1 }, dtype: dtype);
                var grads1 = tf.constant(new[] { 0.01, 0.01 }, dtype: dtype);
                var optimizer = tf.train.GradientDescentOptimizer(3.0f);
                var grads_and_vars = new[] {
                    Tuple.Create(grads0, var0 as IVariableV1),
                    Tuple.Create(grads1, var1 as IVariableV1)
                };
                var sgd_op = optimizer.apply_gradients(grads_and_vars);

                var global_variables = tf.global_variables_initializer();
                sess.run(global_variables);

                var initialVar0 = sess.run(var0);
                var initialVar1 = sess.run(var1);
                // Fetch params to validate initial values
                self.assertAllCloseAccordingToType(new[] { 1.0, 2.0 }, self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0, 4.0 }, self.evaluate<T[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params
                self.assertAllCloseAccordingToType(
                    new[] { 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1 },
                    self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(
                    new[] { 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01 },
                    self.evaluate<T[]>(var1));
                // TODO: self.assertEqual(0, len(optimizer.variables()));
            }
        }

        [TestMethod]
        public void TestBasic()
        {
            //TODO: add np.half
            TestBasic<float>();
            TestBasic<double>();
        }

        private void TestMinimizeResourceVariable<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            // train.GradientDescentOptimizer is V1 only API.
            tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var var0 = tf.Variable(new[,] { { 1.0f, 2.0f } }, dtype: dtype);
                var var1 = tf.Variable(new[] { 3.0 }, dtype: dtype);
                var x = tf.constant(new[,] { { 4.0f }, { 5.0f } }, dtype: dtype);

                var pred = math_ops.matmul(var0, x) + var1;
                var loss = pred * pred;
                var sgd_op = tf.train.GradientDescentOptimizer(1.0f).minimize(loss);

                var global_variables = tf.global_variables_initializer();
                sess.run(global_variables);

                sess.run(new[] { var0, var1 });
                // Fetch params to validate initial values
                self.assertAllCloseAccordingToType<T>(new[,] { { 1.0, 2.0 } }, self.evaluate<T[,]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0 }, self.evaluate<T[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params
                var np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0;
                var np_grad = 2 * np_pred;
                self.assertAllCloseAccordingToType(
                    new[,] { { 1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0 } },
                    self.evaluate<T[,]>(var0));
                self.assertAllCloseAccordingToType(
                    new[] { 3.0 - np_grad },
                    self.evaluate<T[]>(var1));
            }
        }

        [TestMethod]
        public void TestMinimizeResourceVariable()
        {
            //TODO: add np.half
            TestMinimizeResourceVariable<float>();
            TestMinimizeResourceVariable<double>();
        }

        private void TestTensorLearningRate<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            // train.GradientDescentOptimizer is V1 only API.
            tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var var0 = tf.Variable(new[] { 1.0, 2.0 }, dtype: dtype);
                var var1 = tf.Variable(new[] { 3.0, 4.0 }, dtype: dtype);
                var grads0 = tf.constant(new[] { 0.1, 0.1 }, dtype: dtype);
                var grads1 = tf.constant(new[] { 0.01, 0.01 }, dtype: dtype);
                var lrate = constant_op.constant(3.0);
                var grads_and_vars = new[] {
                    Tuple.Create(grads0, var0 as IVariableV1),
                    Tuple.Create(grads1, var1 as IVariableV1)
                };
                var sgd_op = tf.train.GradientDescentOptimizer(lrate)
                    .apply_gradients(grads_and_vars);

                var global_variables = tf.global_variables_initializer();
                sess.run(global_variables);

                var initialVar0 = sess.run(var0);
                var initialVar1 = sess.run(var1);
                // Fetch params to validate initial values
                self.assertAllCloseAccordingToType(new[] { 1.0, 2.0 }, self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0, 4.0 }, self.evaluate<T[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params
                self.assertAllCloseAccordingToType(
                    new[] { 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1 },
                    self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(
                    new[] { 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01 },
                    self.evaluate<T[]>(var1));
                // TODO: self.assertEqual(0, len(optimizer.variables()));
            }
        }

        [TestMethod]
        public void TestTensorLearningRate()
        {
            //TODO: add np.half
            TestTensorLearningRate<float>();
            TestTensorLearningRate<double>();
        }

        public void TestGradWrtRef<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            var graph = tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var opt = tf.train.GradientDescentOptimizer(3.0f);
                var values = new[] { 1.0, 3.0 };
                var vars_ = values.Select(
                        v => tf.Variable(new[] { v }, dtype: dtype) as IVariableV1
                    ).ToList();
                var grads_and_vars = opt.compute_gradients(tf.add(vars_[0], vars_[1]), vars_);
                sess.run(tf.global_variables_initializer());
                foreach (var (grad, _) in grads_and_vars)
                    self.assertAllCloseAccordingToType(new[] { 1.0 }, self.evaluate<T[]>(grad));

            }
        }

        [TestMethod]
        public void TestGradWrtRef()
        {
            TestGradWrtRef<float>();
            TestGradWrtRef<double>();
        }

        public void TestWithGlobalStep<T>() where T : struct
        {
            var dtype = GetTypeForNumericType<T>();

            tf.Graph().as_default();
            using (var sess = self.cached_session())
            {
                var global_step = tf.Variable(0, trainable: false);
                var var0 = tf.Variable(new[] { 1.0, 2.0 }, dtype: dtype);
                var var1 = tf.Variable(new[] { 3.0, 4.0 }, dtype: dtype);
                var grads0 = tf.constant(new[] { 0.1, 0.1 }, dtype: dtype);
                var grads1 = tf.constant(new[] { 0.01, 0.01 }, dtype: dtype);
                var grads_and_vars = new[] {
                    Tuple.Create(grads0, var0 as IVariableV1),
                    Tuple.Create(grads1, var1 as IVariableV1)
                };
                var sgd_op = tf.train.GradientDescentOptimizer(3.0f)
                    .apply_gradients(grads_and_vars, global_step: global_step);

                sess.run(tf.global_variables_initializer());
                // Fetch params to validate initial values
                self.assertAllCloseAccordingToType(new[] { 1.0, 2.0 }, self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0, 4.0 }, self.evaluate<T[]>(var1));
                // Run 1 step of sgd
                sgd_op.run();
                // Validate updated params and global_step
                self.assertAllCloseAccordingToType(new[] { 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1 }, self.evaluate<T[]>(var0));
                self.assertAllCloseAccordingToType(new[] { 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01 }, self.evaluate<T[]>(var1));
                Assert.AreEqual(1, self.evaluate<int>(global_step));
            }

        }

        [TestMethod]
        public void TestWithGlobalStep()
        {
            TestWithGlobalStep<float>();
            TestWithGlobalStep<double>();
        }
    }
}
