using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class InitializerArgs
    {
        public TensorShape Shape { get; set; }
        public TF_DataType DType { get; set; }
        public bool? VerifyShape { get; set; } = null;
    }
}
