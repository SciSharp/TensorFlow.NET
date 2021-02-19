using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Text;

namespace Tensorflow.Keras
{
    public class TextApi
    {
        public Tensorflow.Keras.Text.Tokenizer Tokenizer(
                int num_words = -1,
                string filters = DefaultFilter,
                bool lower = true,
                char split = ' ',
                bool char_level = false,
                string oov_token = null,
                Func<string, IEnumerable<string>> analyzer = null)
        {
            return new Keras.Text.Tokenizer(num_words, filters, lower, split, char_level, oov_token, analyzer);
        }

        public static IEnumerable<string> text_to_word_sequence(string text, string filters = DefaultFilter, bool lower = true, char split = ' ')
        {
            if (lower)
            {
                text = text.ToLower();
            }
            var newText = new String(text.Where(c => !filters.Contains(c)).ToArray());
            return newText.Split(split);
        }

        private const string DefaultFilter = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n";
    }
}
