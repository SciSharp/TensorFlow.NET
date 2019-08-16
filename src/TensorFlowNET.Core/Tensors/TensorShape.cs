using NumSharp;
using System;
using System.Linq;

namespace Tensorflow
{
    /// <summary>
    /// Represents the shape of a `Tensor`.
    /// </summary>
    public class TensorShape
    {
        private Shape shape;
        public int[] dims => shape.Dimensions;
        public int ndim => shape.NDim;
        public int size => shape.Size;

        public TensorShape(TensorShapeProto proto)
        {
            if (proto.UnknownRank) return;

            shape.reshape(proto.Dim.Select(x => (int)x.Size).ToArray());
        }

        public TensorShape(params int[] dims)
        {
            shape = new Shape(dims);
        }

        public TensorShape this[Slice slice]
        {
            get
            {
                return new TensorShape(dims.Skip(slice.Start.Value)
                    .Take(slice.Length.Value)
                    .ToArray());
            }
        }

        /// <summary>
        /// Returns True iff `self` is fully defined in every dimension.
        /// </summary>
        /// <returns></returns>
        public bool is_fully_defined()
        {
            return dims != null && dims.Count(x => x < 1) == 0;
        }

        public bool is_compatible_with(TensorShape shape2)
        {
            throw new NotImplementedException("TensorShape is_compatible_with");
        }

        public TensorShape with_rank_at_least(int rank)
        {
            if (rank != ndim)
                throw new ValueError($"Shape {this} must have rank at least {rank}");
            else
                return this;
        }

        /// <summary>
        /// Returns the concatenation of the dimension in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TensorShape concatenate(int[] other_)
        {
            var other = new TensorShape(other_);

            if (ndim < 0 || other.ndim < 0)
                return new TensorShape();
            else
            {
                var concatenate_dims = new int[ndim + other.ndim];
                for (int i = 0; i < ndim; i++)
                    concatenate_dims[i] = dims[i];

                for (int i = 0; i < other.ndim; i++)
                    concatenate_dims[ndim + i] = other.dims[i];

                return new TensorShape(concatenate_dims);
            }
        }

        public static implicit operator TensorShape(Shape shape) => new TensorShape(shape.Dimensions);
        public static implicit operator Shape(TensorShape shape) => new Shape(shape.dims);
        public static implicit operator TensorShape(int[] dims) => new TensorShape(dims);
        public static implicit operator int[](TensorShape shape) => shape.dims;
        public static implicit operator TensorShape((int, int) dims) => new TensorShape(dims.Item1, dims.Item2);
        public static implicit operator TensorShape((int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3);
        public static implicit operator TensorShape((int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);
    }
}
