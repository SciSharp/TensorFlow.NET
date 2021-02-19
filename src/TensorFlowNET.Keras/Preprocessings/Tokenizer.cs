using NumSharp;
using Serilog.Debugging;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Data.SqlTypes;
using System.Linq;
using System.Net.Sockets;
using System.Text;

namespace Tensorflow.Keras.Text
{
    /// <summary>
    /// Text tokenization API.
    /// This class allows to vectorize a text corpus, by turning each text into either a sequence of integers 
    /// (each integer being the index of a token in a dictionary) or into a vector where the coefficient for 
    /// each token could be binary, based on word count, based on tf-idf...
    /// </summary>
    /// <remarks>
    /// This code is a fairly straight port of the Python code for Keras text preprocessing found at:
    /// https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
    /// </remarks>
    public class Tokenizer
    {
        private readonly int num_words;
        private readonly string filters;
        private readonly bool lower;
        private readonly char split;
        private readonly bool char_level;
        private readonly string oov_token;
        private readonly Func<string, IEnumerable<string>> analyzer;

        private int document_count = 0;

        private Dictionary<string, int> word_docs = new Dictionary<string, int>();
        private Dictionary<string, int> word_counts = new Dictionary<string, int>();

        public Dictionary<string, int> word_index = null;
        public Dictionary<int, string> index_word = null;

        private Dictionary<int, int> index_docs = null;

        public Tokenizer(
            int num_words = -1,
            string filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
            bool lower = true,
            char split = ' ',
            bool char_level = false,
            string oov_token = null,
            Func<string, IEnumerable<string>> analyzer = null)
        {
            this.num_words = num_words;
            this.filters = filters;
            this.lower = lower;
            this.split = split;
            this.char_level = char_level;
            this.oov_token = oov_token;
            this.analyzer = analyzer != null ? analyzer : (text) => TextApi.text_to_word_sequence(text, filters, lower, split);
        }

        /// <summary>
        /// Updates internal vocabulary based on a list of texts. 
        /// </summary>
        /// <param name="texts">A list of strings, each containing one or more tokens.</param>
        /// <remarks>Required before using texts_to_sequences or texts_to_matrix.</remarks>
        public void fit_on_texts(IEnumerable<string> texts)
        {
            foreach (var text in texts)
            {
                IEnumerable<string> seq = null;

                document_count += 1;
                if (char_level)
                {
                    throw new NotImplementedException("char_level == true");
                }
                else
                {
                    seq = analyzer(lower ? text.ToLower() : text);
                }

                foreach (var w in seq)
                {
                    var count = 0;
                    word_counts.TryGetValue(w, out count);
                    word_counts[w] = count + 1;
                }

                foreach (var w in new HashSet<string>(seq))
                {
                    var count = 0;
                    word_docs.TryGetValue(w, out count);
                    word_docs[w] = count + 1;
                }
            }

            var wcounts = word_counts.AsEnumerable().ToList();
            wcounts.Sort((kv1, kv2) => -kv1.Value.CompareTo(kv2.Value));    // Note: '-' gives us descending order.

            var sorted_voc = (oov_token == null) ? new List<string>() : new List<string>() { oov_token };
            sorted_voc.AddRange(word_counts.Select(kv => kv.Key));

            if (num_words > 0 - 1)
            {
                sorted_voc = sorted_voc.Take<string>((oov_token == null) ? num_words : num_words + 1).ToList();
            }

            word_index = new Dictionary<string, int>(sorted_voc.Count);
            index_word = new Dictionary<int, string>(sorted_voc.Count);
            index_docs = new Dictionary<int, int>(word_docs.Count);

            for (int i = 0; i < sorted_voc.Count; i++)
            {
                word_index.Add(sorted_voc[i], i + 1);
                index_word.Add(i + 1, sorted_voc[i]);
            }

            foreach (var kv in word_docs)
            {
                var idx = -1;
                if (word_index.TryGetValue(kv.Key, out idx))
                {
                    index_docs.Add(idx, kv.Value);
                }
            }
        }

        /// <summary>
        /// Updates internal vocabulary based on a list of texts. 
        /// </summary>
        /// <param name="texts">A list of list of strings, each containing one token.</param>
        /// <remarks>Required before using texts_to_sequences or texts_to_matrix.</remarks>
        public void fit_on_texts(IEnumerable<IEnumerable<string>> texts)
        {
            foreach (var seq in texts)
            {
                foreach (var w in seq.Select(s => lower ? s.ToLower() : s))
                {
                    var count = 0;
                    word_counts.TryGetValue(w, out count);
                    word_counts[w] = count + 1;
                }

                foreach (var w in new HashSet<string>(word_counts.Keys))
                {
                    var count = 0;
                    word_docs.TryGetValue(w, out count);
                    word_docs[w] = count + 1;
                }
            }

            var wcounts = word_counts.AsEnumerable().ToList();
            wcounts.Sort((kv1, kv2) => -kv1.Value.CompareTo(kv2.Value));

            var sorted_voc = (oov_token == null) ? new List<string>() : new List<string>() { oov_token };
            sorted_voc.AddRange(word_counts.Select(kv => kv.Key));

            if (num_words > 0 - 1)
            {
                sorted_voc = sorted_voc.Take<string>((oov_token == null) ? num_words : num_words + 1).ToList();
            }

            word_index = new Dictionary<string, int>(sorted_voc.Count);
            index_word = new Dictionary<int, string>(sorted_voc.Count);
            index_docs = new Dictionary<int, int>(word_docs.Count);

            for (int i = 0; i < sorted_voc.Count; i++)
            {
                word_index.Add(sorted_voc[i], i + 1);
                index_word.Add(i + 1, sorted_voc[i]);
            }

            foreach (var kv in word_docs)
            {
                var idx = -1;
                if (word_index.TryGetValue(kv.Key, out idx))
                {
                    index_docs.Add(idx, kv.Value);
                }
            }
        }

        /// <summary>
        /// Updates internal vocabulary based on a list of sequences.
        /// </summary>
        /// <param name="sequences"></param>
        /// <remarks>Required before using sequences_to_matrix (if fit_on_texts was never called).</remarks>
        public void fit_on_sequences(IEnumerable<int[]> sequences)
        {
            throw new NotImplementedException("fit_on_sequences");
        }

        /// <summary>
        /// Transforms each string in texts to a sequence of integers.
        /// </summary>
        /// <param name="texts"></param>
        /// <returns></returns>
        /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
        public IList<int[]> texts_to_sequences(IEnumerable<string> texts)
        {
            return texts_to_sequences_generator(texts).ToArray();
        }

        /// <summary>
        /// Transforms each token in texts to a sequence of integers.
        /// </summary>
        /// <param name="texts"></param>
        /// <returns></returns>
        /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
        public IList<int[]> texts_to_sequences(IEnumerable<IEnumerable<string>> texts)
        {
            return texts_to_sequences_generator(texts).ToArray();
        }

        public IEnumerable<int[]> texts_to_sequences_generator(IEnumerable<string> texts)
        {
            int oov_index = -1;
            var _ = (oov_token != null) && word_index.TryGetValue(oov_token, out oov_index);

            return texts.Select(text =>
            {
                IEnumerable<string> seq = null;

                if (char_level)
                {
                    throw new NotImplementedException("char_level == true");
                }
                else
                {
                    seq = analyzer(lower ? text.ToLower() : text);
                }

                return ConvertToSequence(oov_index, seq).ToArray();
            });
        }

        public IEnumerable<int[]> texts_to_sequences_generator(IEnumerable<IEnumerable<string>> texts)
        {
            int oov_index = -1;
            var _ = (oov_token != null) && word_index.TryGetValue(oov_token, out oov_index);
            return texts.Select(seq => ConvertToSequence(oov_index, seq).ToArray());
        }

        private List<int> ConvertToSequence(int oov_index, IEnumerable<string> seq)
        {
            var vect = new List<int>();
            foreach (var w in seq.Select(s => lower ? s.ToLower() : s))
            {
                var i = -1;
                if (word_index.TryGetValue(w, out i))
                {
                    if (num_words != -1 && i >= num_words)
                    {
                        if (oov_index != -1)
                        {
                            vect.Add(oov_index);
                        }
                    }
                    else
                    {
                        vect.Add(i);
                    }
                }
                else if (oov_index != -1)
                {
                    vect.Add(oov_index);
                }
            }

            return vect;
        }

        /// <summary>
        /// Transforms each sequence into a list of text.
        /// </summary>
        /// <param name="sequences"></param>
        /// <returns>A list of texts(strings)</returns>
        /// <remarks>Only top num_words-1 most frequent words will be taken into account.Only words known by the tokenizer will be taken into account.</remarks>
        public IList<string> sequences_to_texts(IEnumerable<int[]> sequences)
        {
            return sequences_to_texts_generator(sequences).ToArray();
        }

        public IEnumerable<string> sequences_to_texts_generator(IEnumerable<IList<int>> sequences)
        {
            int oov_index = -1;
            var _ = (oov_token != null) && word_index.TryGetValue(oov_token, out oov_index);

            return sequences.Select(seq =>
            {

                var bldr = new StringBuilder();
                for (var i = 0; i < seq.Count; i++)
                {
                    if (i > 0) bldr.Append(' ');

                    string word = null;
                    if (index_word.TryGetValue(seq[i], out word))
                    {
                        if (num_words != -1 && i >= num_words)
                        {
                            if (oov_index != -1)
                            {
                                bldr.Append(oov_token);
                            }
                        }
                        else
                        {
                            bldr.Append(word);
                        }
                    }
                    else if (oov_index != -1)
                    {
                        bldr.Append(oov_token);
                    }
                }

                return bldr.ToString();
            });
        }

        /// <summary>
        /// Convert a list of texts to a Numpy matrix.
        /// </summary>
        /// <param name="texts">A sequence of strings containing one or more tokens.</param>
        /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
        /// <returns></returns>
        public NDArray texts_to_matrix(IEnumerable<string> texts, string mode = "binary")
        {
            return sequences_to_matrix(texts_to_sequences(texts), mode);
        }

        /// <summary>
        /// Convert a list of texts to a Numpy matrix.
        /// </summary>
        /// <param name="texts">A sequence of lists of strings, each containing one token.</param>
        /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
        /// <returns></returns>
        public NDArray texts_to_matrix(IEnumerable<IList<string>> texts, string mode = "binary")
        {
            return sequences_to_matrix(texts_to_sequences(texts), mode);
        }

        /// <summary>
        /// Converts a list of sequences into a Numpy matrix.
        /// </summary>
        /// <param name="sequences">A sequence of lists of integers, encoding tokens.</param>
        /// <param name="mode">One of "binary", "count", "tfidf", "freq".</param>
        /// <returns></returns>
        public NDArray sequences_to_matrix(IEnumerable<IList<int>> sequences, string mode = "binary")
        {
            if (!modes.Contains(mode)) throw new InvalidArgumentError($"Unknown vectorization mode: {mode}");
            var word_count = 0;

            if (num_words == -1)
            {
                if (word_index != null)
                {
                    word_count = word_index.Count + 1;
                }
                else
                {
                    throw new InvalidOperationException("Specifya dimension ('num_words' arugment), or fit on some text data first.");
                }
            }
            else
            {
                word_count = num_words;
            }

            if (mode == "tfidf" && this.document_count == 0)
            {
                throw new InvalidOperationException("Fit the Tokenizer on some text data before using the 'tfidf' mode.");
            }

            var x = np.zeros(sequences.Count(), word_count);

            for (int i = 0; i < sequences.Count(); i++)
            {
                var seq = sequences.ElementAt(i);
                if (seq == null || seq.Count == 0)
                    continue;

                var counts = new Dictionary<int, int>();

                var seq_length = seq.Count;

                foreach (var j in seq)
                {
                    if (j >= word_count)
                        continue;
                    var count = 0;
                    counts.TryGetValue(j, out count);
                    counts[j] = count + 1;
                }

                if (mode == "count")
                {
                    foreach (var kv in counts)
                    {
                        var j = kv.Key;
                        var c = kv.Value;
                        x[i, j] = c;
                    }
                }
                else if (mode == "freq")
                {
                    foreach (var kv in counts)
                    {
                        var j = kv.Key;
                        var c = kv.Value;
                        x[i, j] = ((double)c) / seq_length;
                    }
                }
                else if (mode == "binary")
                {
                    foreach (var kv in counts)
                    {
                        var j = kv.Key;
                        var c = kv.Value;
                        x[i, j] = 1;
                    }
                }
                else if (mode == "tfidf")
                {
                    foreach (var kv in counts)
                    {
                        var j = kv.Key;
                        var c = kv.Value;
                        var id = 0;
                        var _ = index_docs.TryGetValue(j, out id);
                        var tf = 1 + np.log(c);
                        var idf = np.log(1 + document_count / (1 + id));
                        x[i, j] = tf * idf;
                    }
                }
            }

            return x;
        }

        private string[] modes = new string[] { "binary", "count", "tfidf", "freq" };
    }
}
