using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Text.Tokenizers;

namespace Tensorflow.Text
{
    public class TextInterface
    {
        public ITokenizer WhitespaceTokenizer()
            => new WhitespaceTokenizer();

        public Tensor wordshape(Tensor input, WordShape pattern, string name = null)
            => TextOps.wordshape(input, pattern, name: name);

        /// <summary>
        /// Create a tensor of n-grams based on the input data `data`.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="width"></param>
        /// <param name="axis"></param>
        /// <param name="reduction_type"></param>
        /// <param name="string_separator"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor ngrams(Tensor input, int width,
            int axis = -1,
            Reduction reduction_type = Reduction.None,
            string string_separator = " ",
            string name = null)
            => TextOps.ngrams(input, width,
                axis: axis,
                reduction_type: reduction_type,
                string_separator: string_separator,
                name: name);
    }
}
