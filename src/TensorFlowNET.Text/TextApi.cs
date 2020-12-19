using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Text;

namespace Tensorflow
{
    public class TextApi
    {
        public static TextInterface text { get; } = new TextInterface();
    }
}
