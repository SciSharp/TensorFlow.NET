using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class InputList
    {
        public Tensor[] _inputs;

        public InputList(Tensor[] inputs)
        {
            _inputs = inputs;
        }
    }
}
