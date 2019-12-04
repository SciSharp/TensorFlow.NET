using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Dimension
    {
        int _value;
        public int value => _value;

        public Dimension(int value)
        {
            _value = value;
        }

        public Dimension merge_with(Dimension other)
        {
            if (_value == -1)
                return new Dimension(other.value);
            else
                return new Dimension(_value);
        }

        public static implicit operator Dimension(int value)
            => new Dimension(value);

        public static implicit operator int(Dimension dimension)
            => dimension.value;

        public override string ToString() => $"Dimension({_value})";
    }
}
