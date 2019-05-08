using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Estimator;

namespace TensorFlowNET.Examples.Utility
{
    public class CoNLLDataset : IEnumerable
    {
        static Dictionary<string, int> vocab_chars;
        static Dictionary<string, int> vocab_words;

        List<Tuple<int[], int>> _elements;
        HyperParams _hp;

        public CoNLLDataset(string path, HyperParams hp)
        {
            if (vocab_chars == null)
                vocab_chars = load_vocab(hp.filepath_chars);

            if (vocab_words == null)
                vocab_words = load_vocab(hp.filepath_words);

            var lines = File.ReadAllLines(path);

            foreach (var l in lines)
            {
                string line = l.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("-DOCSTART-"))
                {

                }
                else
                {
                    var ls = line.Split(' ');
                    // process word
                    var word = processing_word(ls[0]);
                }
            }
        }

        private (int[], int) processing_word(string word)
        {
            var char_ids = word.ToCharArray().Select(x => vocab_chars[x.ToString()]).ToArray();

            // 1. preprocess word
            if (true) // lowercase
                word = word.ToLower();
            if (false) // isdigit
                word = "$NUM$";

            // 2. get id of word
            int id = vocab_words.GetValueOrDefault(word, vocab_words["$UNK$"]);
            
            return (char_ids, id);
        }

        private Dictionary<string, int> load_vocab(string filename)
        {
            var dict = new Dictionary<string, int>();
            int i = 0;
            File.ReadAllLines(filename)
                .Select(x => dict[x] = i++)
                .Count();
            return dict;
        }

        public IEnumerator GetEnumerator()
        {
            return _elements.GetEnumerator();
        }
    }
}
