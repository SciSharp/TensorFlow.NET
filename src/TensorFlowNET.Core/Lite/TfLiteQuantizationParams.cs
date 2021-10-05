using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Lite
{
    public struct TfLiteQuantizationParams
    {
        public float scale;
        public int zero_point;
    }
}
