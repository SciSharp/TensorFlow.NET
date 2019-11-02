using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public object[] Flatten()
        {
            return new Tensor[] { this };
        }
    }
}
