using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.Keras.Datasets;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class PreprocessingTests : EagerModeTestBase
    {
        private readonly string[] texts = new string[] {
                "It was the best of times, it was the worst of times.",
                "this is a new dawn, an era to follow the previous era. It can not be said to start anew.",
                "It was the best of times, it was the worst of times.",
                "this is a new dawn, an era to follow the previous era.",
                };
        private const string OOV = "<OOV>";

        [TestMethod]
        public void TokenizeWithNoOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(lower: true);
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(23, tokenizer.word_index.Count);

            Assert.AreEqual(7, tokenizer.word_index["worst"]);
            Assert.AreEqual(12, tokenizer.word_index["dawn"]);
            Assert.AreEqual(16, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void TokenizeWithOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(lower: true, oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(24, tokenizer.word_index.Count);

            Assert.AreEqual(1,  tokenizer.word_index[OOV]);
            Assert.AreEqual(8,  tokenizer.word_index["worst"]);
            Assert.AreEqual(13, tokenizer.word_index["dawn"]);
            Assert.AreEqual(17, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void PadSequencesWithDefaults()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(lower: true, oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences);

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(20, padded.shape[1]);

            var firstRow = padded[0];
            var secondRow = padded[1];

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 17].GetInt32());
            for (var i = 0; i < 8; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["previous"], padded[1, 10].GetInt32());
            for (var i = 0; i < 20; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPrePaddingTrunc()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(lower: true, oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences,maxlen:15);

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(15, padded.shape[1]);

            var firstRow = padded[0];
            var secondRow = padded[1];

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 12].GetInt32());
            for (var i = 0; i < 3; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["previous"], padded[1, 5].GetInt32());
            for (var i = 0; i < 15; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }

        [TestMethod]
        public void PadSequencesPostPaddingTrunc()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(lower: true, oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            var padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen: 15, padding: "post", truncating: "post");

            Assert.AreEqual(4, padded.shape[0]);
            Assert.AreEqual(15, padded.shape[1]);

            var firstRow = padded[0];
            var secondRow = padded[1];

            Assert.AreEqual(tokenizer.word_index["worst"], padded[0, 9].GetInt32());
            for (var i = 12; i < 15; i++)
                Assert.AreEqual(0, padded[0, i].GetInt32());
            Assert.AreEqual(tokenizer.word_index["previous"], padded[1, 10].GetInt32());
            for (var i = 0; i < 15; i++)
                Assert.AreNotEqual(0, padded[1, i].GetInt32());
        }
    }
}
