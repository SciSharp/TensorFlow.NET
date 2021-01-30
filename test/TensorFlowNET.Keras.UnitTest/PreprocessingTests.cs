using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.Keras.Datasets;
using Microsoft.Extensions.DependencyInjection;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class PreprocessingTests : EagerModeTestBase
    {
        private readonly string[] texts = new string[] {
                "It was the best of times, it was the worst of times.",
                "Mr and Mrs Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
                "It was the best of times, it was the worst of times.",
                "Mr and Mrs Dursley of number four, Privet Drive.",
                };

        private readonly string[][] tokenized_texts = new string[][] {
                new string[] {"It","was","the","best","of","times","it","was","the","worst","of","times"},
                new string[] {"mr","and","mrs","dursley","of","number","four","privet","drive","were","proud","to","say","that","they","were","perfectly","normal","thank","you","very","much"},
                new string[] {"It","was","the","best","of","times","it","was","the","worst","of","times"},
                new string[] {"mr","and","mrs","dursley","of","number","four","privet","drive"},
                };

        private readonly string[] processed_texts = new string[] {
                "it was the best of times it was the worst of times",
                "mr and mrs dursley of number four privet drive were proud to say that they were perfectly normal thank you very much",
                "it was the best of times it was the worst of times",
                "mr and mrs dursley of number four privet drive",
                };

        private const string OOV = "<OOV>";

        [TestMethod]
        public void TokenizeWithNoOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            Assert.AreEqual(7, tokenizer.word_index["worst"]);
            Assert.AreEqual(12, tokenizer.word_index["number"]);
            Assert.AreEqual(16, tokenizer.word_index["were"]);
        }

        [TestMethod]
        public void TokenizeWithNoOOV_Tkn()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            Assert.AreEqual(7, tokenizer.word_index["worst"]);
            Assert.AreEqual(12, tokenizer.word_index["number"]);
            Assert.AreEqual(16, tokenizer.word_index["were"]);
        }

        [TestMethod]
        public void TokenizeWithOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(28, tokenizer.word_index.Count);

            Assert.AreEqual(1,  tokenizer.word_index[OOV]);
            Assert.AreEqual(8,  tokenizer.word_index["worst"]);
            Assert.AreEqual(13, tokenizer.word_index["number"]);
            Assert.AreEqual(17, tokenizer.word_index["were"]);
        }

        [TestMethod]
        public void TokenizeWithOOV_Tkn()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            Assert.AreEqual(28, tokenizer.word_index.Count);

            Assert.AreEqual(1, tokenizer.word_index[OOV]);
            Assert.AreEqual(8, tokenizer.word_index["worst"]);
            Assert.AreEqual(13, tokenizer.word_index["number"]);
            Assert.AreEqual(17, tokenizer.word_index["were"]);
        }

        [TestMethod]
        public void TokenizeTextsToSequences()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            Assert.AreEqual(tokenizer.word_index["worst"], sequences[0][9]);
            Assert.AreEqual(tokenizer.word_index["proud"], sequences[1][10]);
        }

        [TestMethod]
        public void TokenizeTextsToSequences_Tkn()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            var sequences = tokenizer.texts_to_sequences(tokenized_texts);
            Assert.AreEqual(4, sequences.Count);

            Assert.AreEqual(tokenizer.word_index["worst"], sequences[0][9]);
            Assert.AreEqual(tokenizer.word_index["proud"], sequences[1][10]);
        }

        [TestMethod]
        public void TokenizeTextsToSequencesAndBack()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            var processed = tokenizer.sequences_to_texts(sequences);

            Assert.AreEqual(4, processed.Count);

            for (var i = 0; i < processed.Count; i++)
                Assert.AreEqual(processed_texts[i], processed[i]);
        }

        [TestMethod]
        public void TokenizeTextsToSequencesAndBack_Tkn1()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            // Use the list version, where the tokenization has already been done.
            var sequences = tokenizer.texts_to_sequences(tokenized_texts);
            Assert.AreEqual(4, sequences.Count);

            var processed = tokenizer.sequences_to_texts(sequences);

            Assert.AreEqual(4, processed.Count);

            for (var i = 0; i < processed.Count; i++)
                Assert.AreEqual(processed_texts[i], processed[i]);
        }

        [TestMethod]
        public void TokenizeTextsToSequencesAndBack_Tkn2()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            var processed = tokenizer.sequences_to_texts(sequences);

            Assert.AreEqual(4, processed.Count);

            for (var i = 0; i < processed.Count; i++)
                Assert.AreEqual(processed_texts[i], processed[i]);
        }

        [TestMethod]
        public void TokenizeTextsToSequencesAndBack_Tkn3()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            // Use the list version, where the tokenization has already been done.
            var sequences = tokenizer.texts_to_sequences(tokenized_texts);
            Assert.AreEqual(4, sequences.Count);

            var processed = tokenizer.sequences_to_texts(sequences);

            Assert.AreEqual(4, processed.Count);

            for (var i = 0; i < processed.Count; i++)
                Assert.AreEqual(processed_texts[i], processed[i]);
        }
        [TestMethod]
        public void TokenizeTextsToSequencesWithOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            Assert.AreEqual(tokenizer.word_index["worst"], sequences[0][9]);
            Assert.AreEqual(tokenizer.word_index["proud"], sequences[1][10]);

            for (var i = 0; i < sequences.Count; i++)
                for (var j = 0; j < sequences[i].Length; j++)
                Assert.AreNotEqual(tokenizer.word_index[OOV], sequences[i][j]);
        }

        [TestMethod]
        public void TokenizeTextsToSequencesWithOOVPresent()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV, num_words:20);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            Assert.AreEqual(tokenizer.word_index["worst"], sequences[0][9]);
            Assert.AreEqual(tokenizer.word_index["proud"], sequences[1][10]);

            var oov_count = 0;
            for (var i = 0; i < sequences.Count; i++)
                for (var j = 0; j < sequences[i].Length; j++)
                    if (tokenizer.word_index[OOV] == sequences[i][j])
                        oov_count += 1;

            Assert.AreEqual(9, oov_count);
        }

        [TestMethod]
        public void PadSequencesWithDefaults()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences);

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(22, padded.shape[1]);

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 19].GetInt32());
            for (var i = 0; i < 8; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["proud"], padded[1, 10].GetInt32());
            for (var i = 0; i < 20; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPrePaddingTrunc()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences,maxlen:15);

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(15, padded.shape[1]);

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 12].GetInt32());
            for (var i = 0; i < 3; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["proud"], padded[1, 3].GetInt32());
            for (var i = 0; i < 15; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPrePaddingTrunc_Larger()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen: 45);

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(45, padded.shape[1]);

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 42].GetInt32());
            for (var i = 0; i < 33; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["proud"], padded[1, 33].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPostPaddingTrunc()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen: 15, padding: "post", truncating: "post");

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(15, padded.shape[1]);

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 9].GetInt32());
            for (var i = 12; i < 15; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["proud"], padded[1, 10].GetInt32());
            for (var i = 0; i < 15; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPostPaddingTrunc_Larger()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen: 45, padding: "post", truncating: "post");

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(45, padded.shape[1]);

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 9].GetInt32());
            for (var i = 32; i < 45; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["proud"], padded[1, 10].GetInt32());
        }

        [TestMethod]
        public void TextToMatrixBinary()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            var matrix = tokenizer.texts_to_matrix(texts);

            Assert.AreEqual(texts.Length, matrix.shape[0]);

            CompareLists(new double[] { 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, matrix[0].ToArray<double>());
            CompareLists(new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, matrix[1].ToArray<double>());
        }

        [TestMethod]
        public void TextToMatrixCount()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            var matrix = tokenizer.texts_to_matrix(texts, mode:"count");

            Assert.AreEqual(texts.Length, matrix.shape[0]);

            CompareLists(new double[] { 0, 2, 2, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, matrix[0].ToArray<double>());
            CompareLists(new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, matrix[1].ToArray<double>());
        }

        [TestMethod]
        public void TextToMatrixFrequency()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            var matrix = tokenizer.texts_to_matrix(texts, mode: "freq");

            Assert.AreEqual(texts.Length, matrix.shape[0]);

            double t12 = 2.0 / 12.0;
            double o12 = 1.0 / 12.0;
            double t22 = 2.0 / 22.0;
            double o22 = 1.0 / 22.0;

            CompareLists(new double[] { 0, t12, t12, t12, o12, t12, t12, o12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, matrix[0].ToArray<double>());
            CompareLists(new double[] { 0, 0, 0, 0, 0, o22, 0, 0, o22, o22, o22, o22, o22, o22, o22, o22, t22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22, o22 }, matrix[1].ToArray<double>());
        }

        [TestMethod]
        public void TextToMatrixTDIDF()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(27, tokenizer.word_index.Count);

            var matrix = tokenizer.texts_to_matrix(texts, mode: "tfidf");

            Assert.AreEqual(texts.Length, matrix.shape[0]);

            double t1 = 1.1736001944781467;
            double t2 = 0.69314718055994529;
            double t3 = 1.860112299086919;
            double t4 = 1.0986122886681098;
            double t5 = 0.69314718055994529;

            CompareLists(new double[] { 0, t1, t1, t1, t2, 0, t1, t2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, matrix[0].ToArray<double>());
            CompareLists(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, t5, t5, t5, t5, t5, t5, t5, t5, t3, t4, t4, t4, t4, t4, t4, t4, t4, t4, t4, t4 }, matrix[1].ToArray<double>());
        }

        private void CompareLists<T>(IList<T> expected, IList<T> actual)
        {
            Assert.AreEqual(expected.Count, actual.Count);
            for (var i = 0; i < expected.Count; i++)
            {
                Assert.AreEqual(expected[i], actual[i]);
            }
        }

    }
}
