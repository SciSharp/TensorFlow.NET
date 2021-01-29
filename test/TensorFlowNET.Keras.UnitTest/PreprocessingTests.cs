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

        private readonly string[][] tokenized_texts = new string[][] {
                new string[] {"It","was","the","best","of","times","it","was","the","worst","of","times"},
                new string[] {"this","is","a","new","dawn","an","era","to","follow","the","previous","era","It","can","not","be","said","to","start","anew" },
                new string[] {"It","was","the","best","of","times","it","was","the","worst","of","times"},
                new string[] {"this","is","a","new","dawn","an","era","to","follow","the","previous","era" },
                };

        private readonly string[] processed_texts = new string[] {
                "it was the best of times it was the worst of times",
                "this is a new dawn an era to follow the previous era it can not be said to start anew",
                "it was the best of times it was the worst of times",
                "this is a new dawn an era to follow the previous era",
                };

        private const string OOV = "<OOV>";

        [TestMethod]
        public void TokenizeWithNoOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(23, tokenizer.word_index.Count);

            Assert.AreEqual(7, tokenizer.word_index["worst"]);
            Assert.AreEqual(12, tokenizer.word_index["dawn"]);
            Assert.AreEqual(16, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void TokenizeWithNoOOV_Tkn()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            Assert.AreEqual(23, tokenizer.word_index.Count);

            Assert.AreEqual(7, tokenizer.word_index["worst"]);
            Assert.AreEqual(12, tokenizer.word_index["dawn"]);
            Assert.AreEqual(16, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void TokenizeWithOOV()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            tokenizer.fit_on_texts(texts);

            Assert.AreEqual(24, tokenizer.word_index.Count);

            Assert.AreEqual(1,  tokenizer.word_index[OOV]);
            Assert.AreEqual(8,  tokenizer.word_index["worst"]);
            Assert.AreEqual(13, tokenizer.word_index["dawn"]);
            Assert.AreEqual(17, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void TokenizeWithOOV_Tkn()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
            // Use the list version, where the tokenization has already been done.
            tokenizer.fit_on_texts(tokenized_texts);

            Assert.AreEqual(24, tokenizer.word_index.Count);

            Assert.AreEqual(1, tokenizer.word_index[OOV]);
            Assert.AreEqual(8, tokenizer.word_index["worst"]);
            Assert.AreEqual(13, tokenizer.word_index["dawn"]);
            Assert.AreEqual(17, tokenizer.word_index["follow"]);
        }

        [TestMethod]
        public void TokenizeTextsToSequences()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer();
            tokenizer.fit_on_texts(texts);

            var sequences = tokenizer.texts_to_sequences(texts);
            Assert.AreEqual(4, sequences.Count);

            Assert.AreEqual(tokenizer.word_index["worst"], sequences[0][9]);
            Assert.AreEqual(tokenizer.word_index["previous"], sequences[1][10]);
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
            Assert.AreEqual(tokenizer.word_index["previous"], sequences[1][10]);
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
            Assert.AreEqual(tokenizer.word_index["previous"], sequences[1][10]);

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
            Assert.AreEqual(tokenizer.word_index["previous"], sequences[1][10]);

            var oov_count = 0;
            for (var i = 0; i < sequences.Count; i++)
                for (var j = 0; j < sequences[i].Length; j++)
                    if (tokenizer.word_index[OOV] == sequences[i][j])
                        oov_count += 1;

            Assert.AreEqual(5, oov_count);
        }

        [TestMethod]
        public void PadSequencesWithDefaults()
        {
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
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
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
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
            var tokenizer = keras.preprocessing.text.Tokenizer(oov_token: OOV);
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
