using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class Shape
    {
        public int ndim => _dims.Length;
        long[] _dims;
        public long[] dims => _dims;

        public Shape() 
        {
        }

        public Shape(params int[] dims)
            => _dims = dims.Select(x => Convert.ToInt64(x)).ToArray();

        public Shape(params long[] dims)
            => _dims = dims;

        public static implicit operator Shape(int dims)
            => new Shape(dims);

        public static implicit operator Shape(long[] dims)
            => new Shape(dims);

        public static implicit operator Shape(int[] dims)
            => new Shape(dims);

        public static implicit operator Shape((int, int) dims)
            => new Shape(dims.Item1, dims.Item2);

        public static implicit operator Shape((long, long) dims)
            => new Shape(dims.Item1, dims.Item2);

        public static implicit operator Shape((int, int, int) dims)
            => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static implicit operator Shape((long, long, long) dims)
            => new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static implicit operator Shape((int, int, int, int) dims)
            => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static implicit operator Shape((long, long, long, long) dims)
            => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public bool IsEmpty => size == 0;

        public bool IsScalar => ndim == 0;

        public static Shape Scalar
            => new Shape(new long[0]);

        /// <summary>
        ///     Returns the size this shape represents.
        /// </summary>
        public ulong size
        {
            get
            {
                // scalar
                if (ndim == 0) 
                    return 1;

                var computed = 1L;
                for (int i = 0; i < _dims.Length; i++)
                {
                    var val = _dims[i];
                    if (val == 0)
                        return 0;
                    else if (val < 0)
                        continue;
                    computed *= val;
                }

                return (ulong)computed;
            }
        }

        public override bool Equals(object obj)
        {
            if(obj is Shape shape)
            {
                if (shape.ndim != ndim)
                    return false;
                if (Enumerable.SequenceEqual(dims, shape.dims))
                    return true;
            }
            return base.Equals(obj);
        }

        public override string ToString()
        {
            return "(" + string.Join(", ", _dims) + ")";
        }
    }
}
