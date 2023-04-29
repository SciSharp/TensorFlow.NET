using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class AttentionTest : EagerModeTestBase
    {
        #region BaseDenseAttention

        [TestMethod]
        public void test_multi_dim_with_mask()
        {
            // Scores tensor of shape [1, 1, 3]
            var scores = np.array(new[, ,] { { { 1f, 0f, 1f } } }, dtype: np.float32);
            // Value tensor of shape [1, 3, 1]
            var v = np.array(new[, ,] { { { 1.6f }, { 0.7f }, { -0.8f } } }, dtype: np.float32);
            // Scores mask tensor of shape [1, 1, 3]
            var scores_mask = np.array(new[, ,] { { { true, true, false } } }, dtype: np.@bool);
            var _tup_1 = new BaseDenseAttention(new())._apply_scores(scores: scores, value: v, scores_mask: scores_mask);
            var actual = _tup_1.Item1;
            var actual_scores = _tup_1.Item2;
            // Expected softmax scores = softmax(scores) with zeros in positions where
            // v_mask == False.
            // => softmax_scores000 = exp(1)/(exp(1) + exp(0)) = 0.73105857863
            //    softmax_scores001 = exp(0)/(exp(1) + exp(0)) = 0.26894142137
            //    softmax_scores002 = 0
            var expected_scores = np.array(new[, ,] { { { 0.73105857863f, 0.26894142137f, 0f } } }, dtype: np.float32);
            Assert.AreEqual(expected_scores, actual_scores.numpy());
            // Expected tensor of shape [1, 1, 1].
            // expected000 = 0.73105857863 * 1.6 + 0.26894142137 * 0.7 - 0 * 0.8
            //             = 1.35795272077
            //Actually the output is 1.3579528
            var expected = np.array(new[, ,] { { { 1.3579528f } } }, dtype: np.float32);
            Assert.AreEqual(expected, actual.numpy());
        }

        [TestMethod]
        public void test_one_dim_batch_size_two()
        {
            // Scores tensor of shape [2, 1, 1]
            var scores = np.array(new[, ,] { { { 1.1f } }, { { 2.1f } } }, dtype: np.float32);
            // Value tensor of shape [2, 1, 1]
            var v = np.array(new[, ,] { { { 1.6f } }, { { 2.6f } } }, dtype: np.float32);
            // Scpres mask tensor of shape [2, 1, 1]
            var scores_mask = np.array(new[, ,] { { { true } }, { { true } } }, dtype: np.@bool);
            var _tup_1 = new BaseDenseAttention(new())._apply_scores(scores: scores, value: v, scores_mask: scores_mask);
            var actual = _tup_1.Item1;
            var actual_scores = _tup_1.Item2;
            // Expected softmax_scores = [[[1]], [[1]]]
            var expected_scores = np.array(new[, ,] { { { 1f } }, { { 1f } } }, dtype: np.float32);
            Assert.AreEqual(expected_scores, actual_scores.numpy());
            // Expected tensor of shape [2, 1, 1].
            // expected000 = softmax_scores[0, 0] * 1.6 = 1.6
            // expected100 = softmax_scores[1, 0] * 2.6 = 2.6
            var expected = np.array(new[, ,] { { { 1.6f } }, { { 2.6f } } }, dtype: np.float32);
            Assert.AreEqual(expected, actual.numpy());
        }
        #endregion
        // ------------------------------------------------------------------
        #region Attention


        [TestMethod]
        public void test_calculate_scores_multi_dim()
        {
            // Query tensor of shape [1, 2, 4]
            var q = np.array(new[, ,] { {
                { 1f, 1.1f, 1.2f, 1.3f },
                { 2f, 2.1f, 2.2f, 2.3f }
            } }, dtype: np.float32);
            // Key tensor of shape [1, 3, 4]
            var k = np.array(new[, ,] { {
                { 1.5f, 1.6f, 1.7f, 1.8f },
                { 2.5f, 2.6f, 2.7f, 2.8f },
                { 3.5f, 3.6f, 3.7f, 3.8f }
            } }, dtype: np.float32);
            var attention_layer = (Attention)keras.layers.Attention();
            //attention_layer.build(((1, 2, 4), (1, 3, 4)));
            var actual = attention_layer._calculate_scores(query: q, key: k);
            // Expected tensor of shape [1, 2, 3].
            // expected000 = 1.*1.5+1.1*1.6+1.2*1.7+1.3*1.8 = 7.64
            // expected001 = 1.*2.5+1.1*2.6+1.2*2.7+1.3*2.8 = 12.24
            // expected002 = 1.*3.5+1.1*3.6+1.2*3.7+1.3*3.8 = 16.84
            // expected010 = 2.*1.5+2.1*1.6+2.2*1.7+2.3*1.8 = 14.24
            // expected011 = 2.*2.5+2.1*2.6+2.2*2.7+2.3*2.8 = 22.84
            // expected012 = 2.*3.5+2.1*3.6+2.2*3.7+2.3*3.8 = 31.44
            // Actually the output000 is 7.6400003, the output012 is 31.439999
            var expected = np.array(new[, ,] { {
                { 7.6400003f, 12.24f, 16.84f },
                { 14.24f, 22.84f, 31.439999f }
            } }, dtype: np.float32);
            Assert.IsTrue(expected == actual.numpy());
        }

        [TestMethod]
        [Ignore]
        public void test_calculate_scores_multi_dim_concat()
        {
            // Query tensor of shape [1, 2, 4]
            var q = np.array(new[, ,] { {
                { 1f, 1.1f, 1.2f, 1.3f },
                { 2f, 2.1f, 2.2f, 2.3f }
            } }, dtype: np.float32);
            // Key tensor of shape [1, 3, 4]
            var k = np.array(new[, ,] { {
                { 1.5f, 1.6f, 1.7f, 1.8f },
                { 2.5f, 2.6f, 2.7f, 2.8f },
                { 3.5f, 3.6f, 3.7f, 3.8f }
            } }, dtype: np.float32);
            var attention_layer = (Attention)keras.layers.Attention(score_mode: "concat");
            //attention_layer.concat_score_weight = 1;
            attention_layer.concat_score_weight = base_layer_utils.make_variable(new VariableArgs()
            {
                Name = "concat_score_weight",
                Shape = (1),
                DType = TF_DataType.TF_FLOAT,
                Getter = base_layer_utils.make_variable,
                Overwrite = true,
                Initializer = tf.ones_initializer,
                Synchronization = VariableSynchronization.Auto,
                Aggregation = VariableAggregation.None,
                Trainable = true
            });
            //attention_layer.build(((1, 2, 4), (1, 3, 4)));
            //var actual = keras.backend.get_value(attention_layer._calculate_scores(query: q, key: k));
            var actual = attention_layer._calculate_scores(query: q, key: k);
            // pylint:disable=line-too-long
            // expected000 = tanh(1.+1.5) + tanh(1.1+1.6) + tanh(1.2+1.7) + tanh(1.3+1.8) = 3.96753427840
            // expected001 = tanh(1.+2.5) + tanh(1.1+2.6) + tanh(1.2+2.7) + tanh(1.3+2.8) = 3.99558784825
            // expected002 = tanh(1.+3.5) + tanh(1.1+3.6) + tanh(1.2+3.7) + tanh(1.3+3.8) = 3.99940254147
            // expected010 = tanh(2.+1.5) + tanh(2.1+1.6) + tanh(2.2+1.7) + tanh(2.3+1.8) = 3.99558784825
            // expected011 = tanh(2.+2.5) + tanh(2.1+2.6) + tanh(2.2+2.7) + tanh(2.3+2.8) = 3.99940254147
            // expected012 = tanh(2.+3.5) + tanh(2.1+3.6) + tanh(2.2+3.7) + tanh(2.3+3.8) = 3.99991913657
            //Actually the output012 is 3.9999194
            var expected = np.array(new[, ,] { {
                { 3.96753427840f, 3.99558784825f, 3.99940254147f },
                { 3.99558784825f, 3.99940254147f, 3.9999194f }
            } }, dtype: np.float32);
            Assert.AreEqual(expected, actual.numpy());
        }
        #endregion
        // ------------------------------------------------------------------
        #region MultiHeadAttention
        [TestMethod]
        public void test_masked_attention()
        {
            var batch_size = 3;

            var query = keras.Input(shape: (4, 8));
            var value = keras.Input(shape: (2, 8));
            var mask_tensor = keras.Input(shape: (4, 2));
            var attention_layer = keras.layers.MultiHeadAttention(num_heads: 2, key_dim: 2);
            attention_layer.Apply(new Tensor[] { query, value, mask_tensor });

            var from_data = 10 * np.random.randn(batch_size, 4, 8);
            var to_data = 10 * np.random.randn(batch_size, 2, 8);

            var mask_data = np.random.randint(2, size: (batch_size, 4, 2));
            var masked_output_data = attention_layer.Apply(new[] { from_data, to_data, mask_data });

            var null_mask_data = np.ones((batch_size, 4, 2));
            var unmasked_output_data = attention_layer.Apply(new[] { from_data, to_data, null_mask_data });

            Assert.AreNotEqual(masked_output_data, unmasked_output_data);
        }
        #endregion
    }

}
