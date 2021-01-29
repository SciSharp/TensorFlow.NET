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
            wcounts.Sort((kv1, kv2) => -kv1.Value.CompareTo(kv2.Value));

            var sorted_voc = (oov_token == null) ? new List<string>() : new List<string>(){oov_token};
            sorted_voc.AddRange(word_counts.Select(kv => kv.Key));

            if (num_words > 0 -1)
            {
                sorted_voc = sorted_voc.Take<string>((oov_token == null) ? num_words : num_words + 1).ToList();
            }

            word_index = new Dictionary<string, int>(sorted_voc.Count);
            index_word = new Dictionary<int,string>(sorted_voc.Count);
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

        public void fit_on_sequences(IEnumerable<int[]> sequences)
        {
            throw new NotImplementedException("fit_on_sequences");
        }

        public IList<int[]> texts_to_sequences(IEnumerable<string> texts)
        {
            return texts_to_sequences_generator(texts).ToArray();
        }
        public IEnumerable<int[]> texts_to_sequences_generator(IEnumerable<string> texts)
        {
            int oov_index = -1;
            var _ = (oov_token != null) && word_index.TryGetValue(oov_token, out oov_index);

            return texts.Select(text => {

                IEnumerable<string> seq = null;

                if (char_level) 
                {
                    throw new NotImplementedException("char_level == true");
                }
                else
                {
                    seq = analyzer(lower ? text.ToLower() : text);
                }
                var vect = new List<int>();
                foreach (var w in seq)
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
                    else
                    {
                        vect.Add(oov_index);
                    }
                }

                return vect.ToArray();
            });
        }

        public IEnumerable<string> sequences_to_texts(IEnumerable<int[]> sequences)
        {
            throw new NotImplementedException("sequences_to_texts");
        }

        public NDArray sequences_to_matrix(IEnumerable<int[]> sequences)
        {
            throw new NotImplementedException("sequences_to_matrix");
        }
    }
}
