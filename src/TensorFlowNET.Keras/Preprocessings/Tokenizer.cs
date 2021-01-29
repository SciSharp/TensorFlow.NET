using NumSharp;
using Serilog.Debugging;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
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
            this.analyzer = analyzer;
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
        /// Converts a list of sequences into a Numpy matrix.
        /// </summary>
        /// <param name="sequences"></param>
        /// <returns></returns>
        public NDArray sequences_to_matrix(IEnumerable<IList<int>> sequences)
        {
            throw new NotImplementedException("sequences_to_matrix");
        }
    }
}
