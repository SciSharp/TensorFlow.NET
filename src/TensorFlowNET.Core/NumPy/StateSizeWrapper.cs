using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;


namespace Tensorflow.NumPy
{
    // Since state_size in RNN is a single integer or array of integer, so use StateSizeWrapper to hold it
    public class StateSizeWrapper : IEnumerable<int>
    {
        int[] _state_size;
        public int[] state_size => _state_size;

        public StateSizeWrapper(int state_size)
        {
            _state_size = new int[] { state_size };
        }

        public StateSizeWrapper(params int[] state_size)
        {
            _state_size = state_size;
        }
        public StateSizeWrapper(IEnumerable<int> state_size)
        {
            _state_size = state_size.ToArray();
        }

        public static implicit operator StateSizeWrapper(int[] state_size)
            => new StateSizeWrapper(state_size);

        public static implicit operator StateSizeWrapper(int state_size)
            => new StateSizeWrapper(state_size);

        public static implicit operator StateSizeWrapper((int, int) state_size)
            => new StateSizeWrapper(state_size.Item1, state_size.Item2);

        public static implicit operator StateSizeWrapper(List<int> v)
        => new StateSizeWrapper(v);
        public override string ToString()
        {
            return $"{state_size}";
        }

        public int this[int n]
        {
            get => n < 0 ? state_size[state_size.Length + n] : state_size[n];
            set => state_size[n] = value;
        }

        public IEnumerator<int> GetEnumerator()
        {
            return state_size.ToList().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}


