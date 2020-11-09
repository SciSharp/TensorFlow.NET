using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;

namespace Tensorflow
{
    public static class KerasExt
    {
        public static KerasApi Keras(this tensorflow tf)
            => new KerasApi();

        public static KerasApi keras { get; } = new KerasApi();
    }
}
