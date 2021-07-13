using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class Shape
    {
        public int ndim => _dims == null ? -1 : _dims.Length;
        long[] _dims;
        public long[] dims => _dims;

        private Shape() 
        {
        }

        public Shape(TensorShapeProto proto)
        {
            _dims = proto.Dim.Select(x => x.Size).ToArray();
        }

        public void Deconstruct(out long h, out long w)
        {
            h = dims[0];
            w = dims[1];
        }

        public Shape(params int[] dims)
            => _dims = dims?.Select(x => Convert.ToInt64(x))?.ToArray();

        public Shape(params long[] dims)
            => _dims = dims;

        public static implicit operator Shape(int dims)
            => new Shape(dims);

        public static implicit operator Shape(long[] dims)
            => dims == null ? null : new Shape(dims);

        public static implicit operator Shape(int[] dims)
            => dims == null ? null : new Shape(dims);

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

        public static implicit operator int[](Shape shape)
            => shape.dims.Select(x => (int)x).ToArray();

        public static implicit operator long[](Shape shape)
            => shape.dims;

        public bool IsEmpty => size == 0;

        public bool IsScalar => ndim == 0;
        public bool IsNull => _dims == null;

        public bool IsFullyDefined => ndim > -1 && dims.Count(x => x < 1) == 0;

        public static Shape Scalar => new Shape(new long[0]);
        public static Shape Null => new Shape();

        public long this[int n] 
        {
            get => dims[n];
            set => dims[n] = value;
        }

        public Shape this[Slice slice]
        {
            get
            {
                if (!slice.Stop.HasValue)
                    slice.Stop = dims.Length - slice.Start + 1;

                if (slice.Start.HasValue == false || slice.Length.HasValue == false)
                    throw new ArgumentException("Slice must has Start and Length.");

                return new Shape(dims.Skip(slice.Start.Value)
                    .Take(slice.Length.Value)
                    .ToArray());
            }
        }

        /// <summary>
        ///     Returns the size this shape represents.
        /// </summary>
        public long size
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

                return computed;
            }
        }

        public bool is_compatible_with(Shape shape2)
        {
            if (dims != null && shape2.dims != null)
            {
                if (dims.Contains(-1) || shape2.dims.Contains(-1))
                    return true;

                if (size != shape2.size)
                    return false;
            }

            return true;
        }

        public Shape with_rank_at_least(int rank)
        {
            if (ndim < rank)
                throw new ValueError($"Shape {this} must have rank at least {rank}");
            else
                return this;
        }

        public Shape with_rank(int rank)
        {
            return merge_with(unknown_shape(rank: rank));
        }

        /// <summary>
        /// Returns an unknown Shape, optionally with a known rank.
        /// </summary>
        /// <param name="rank"></param>
        /// <returns></returns>
        public Shape unknown_shape(int rank = -1)
        {
            if (rank == -1)
                return Shape.Null;
            else
                return new Shape(Enumerable.Repeat(-1L, rank).ToArray());
        }

        public Shape concatenate(long[] other)
        {
            return concatenate(new Shape(other));
        }

        /// <summary>
        ///     Returns the concatenation of the dimension in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Shape concatenate(Shape other)
        {
            var otherShape = other;

            if (ndim < 0 || otherShape.ndim < 0)
                return Shape.Null;
            else
            {
                var concatenate_dims = new long[ndim + otherShape.ndim];
                for (int i = 0; i < ndim; i++)
                    concatenate_dims[i] = dims[i];

                for (int i = 0; i < otherShape.ndim; i++)
                    concatenate_dims[ndim + i] = otherShape.dims[i];

                return new Shape(concatenate_dims);
            }
        }

        /// <summary>
        /// Returns a `Shape` combining the information in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Shape merge_with(Shape other)
        {
            if (dims == null)
                return other;

            var new_dims = new List<long>();

            foreach (var i in Enumerable.Range(0, ndim))
            {
                var dim = new Dimension(dims[i]);
                var merged = dim.merge_with(new Dimension(other.dims[i]));
                new_dims.Add(merged.value);
            }

            return new Shape(new_dims.ToArray());
        }

        public int[] as_int_list()
        {
            return _dims.Select(x => (int)x).ToArray();
        }

        public void assert_has_rank(int rank)
        {
            if (rank != ndim)
                throw new ValueError(String.Format("Shape {0} must have rank {1}", ndim, rank));
        }

        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case Shape shape1:
                    if (ndim == -1 && shape1.ndim == -1)
                        return false;
                    else if (ndim != shape1.ndim)
                        return false;
                    return Enumerable.SequenceEqual(shape1.dims, dims);
                case long[] shape2:
                    if (ndim != shape2.Length)
                        return false;
                    return Enumerable.SequenceEqual(dims, shape2);
                default:
                    return false;
            }
        }

        public override string ToString()
            => ndim switch
            {
                -1 => "<unknown>",
                0 => "()",
                1 => $"({dims[0]},)",
                _ => $"({string.Join(", ", _dims).Replace("-1", "None")})"
            };
    }
}
