using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public Tensor Pack(object[] sequences)
        {
            return sequences[0] as Tensor;
        }
    }
}
