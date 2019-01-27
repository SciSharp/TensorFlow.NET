using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class InputList : IEnumerable
    {
        public Tensor[] _inputs;
        public Tensor this[int index] => _inputs[index];

        public InputList(Tensor[] inputs)
        {
            _inputs = inputs;
        }

        public IEnumerator GetEnumerator()
        {
            return _inputs.GetEnumerator();
        }
    }
}
