using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Text
{
    public enum WordShape
    {
        HAS_TITLE_CASE,
        IS_UPPERCASE,
        HAS_SOME_PUNCT_OR_SYMBOL,
        IS_NUMERIC_VALUE
    }
}
