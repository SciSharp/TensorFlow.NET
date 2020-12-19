using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Text.Tokenizers
{
    public interface ITokenizer
    {
        Tensor tokenize(Tensor input);
    }
}
