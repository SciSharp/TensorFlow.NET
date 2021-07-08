namespace Tensorflow
{
    public class Dimension
    {
        long _value;
        public long value => _value;

        public Dimension(long value)
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

        public static implicit operator Dimension(long value)
            => new Dimension(value);

        public static implicit operator long(Dimension dimension)
            => dimension.value;

        public override string ToString() => $"Dimension({_value})";
    }
}
