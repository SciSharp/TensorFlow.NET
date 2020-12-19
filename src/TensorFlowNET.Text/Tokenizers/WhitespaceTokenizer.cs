using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Text.Tokenizers
{
    public class WhitespaceTokenizer : ITokenizer
    {
        /// <summary>
        /// Tokenizes a tensor of UTF-8 strings on whitespaces.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor tokenize(Tensor input)
        {
            throw new NotImplementedException("");
        }
    }
}
