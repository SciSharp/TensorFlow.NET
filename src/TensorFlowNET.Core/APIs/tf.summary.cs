using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.IO;
using Tensorflow.Summaries;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Summary summary = new Summary();
        public static Tensor scalar(string name, Tensor tensor) 
            => summary.scalar(name, tensor);
    }
}
