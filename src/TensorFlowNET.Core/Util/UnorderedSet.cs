using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Util
{
    public class UnorderedSet<T> : HashSet<T>
    {
        public UnorderedSet(T[] elements)
        {
            foreach (var el in elements)
                Add(el);
        }

        public bool find(T value)
            => Contains(value);
    }
}
