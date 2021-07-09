using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.NumPy
{
    public class Shape
    {
        public int ndim => _dims.Length;
        long[] _dims;
        public long[] dims => _dims;

        public Shape(params long[] dims)
            => _dims = dims;

        public static implicit operator Shape(int dims)
            => new Shape(dims);

        public static implicit operator Shape(long[] dims)
            => new Shape(dims);

        public static implicit operator Shape(int[] dims)
            => new Shape(dims.Select(x => Convert.ToInt64(x)).ToArray());

        public static implicit operator Shape((long, long) dims)
            => new Shape(dims.Item1, dims.Item2);

        public bool IsSliced => throw new NotImplementedException("");
        public bool IsScalar => throw new NotImplementedException("");
        public bool IsBroadcasted => throw new NotImplementedException("");

        public static Shape Scalar
            => new Shape(new long[0]);

        /// <summary>
        ///     Returns the size this shape represents.
        /// </summary>
        public ulong size
        {
            get
            {
                var computed = 1L;
                for (int i = 0; i < _dims.Length; i++)
                {
                    var val = _dims[i];
                    if (val <= 0)
                        continue;
                    computed *= val;
                }

                return (ulong)computed;
            }
        }

        public bool IsEmpty => throw new NotImplementedException("");

        public override string ToString()
        {
            return "(" + string.Join(", ", _dims) + ")";
        }
    }
}
