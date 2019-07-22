using NumSharp;
using System;
using System.Linq;

namespace Tensorflow
{
    /// <summary>
    /// Represents the shape of a `Tensor`.
    /// </summary>
    public class TensorShape : Shape
    {
        public TensorShape(TensorShapeProto proto)
        {
            if (proto.UnknownRank) return;

            Reshape(proto.Dim.Select(x => (int)x.Size).ToArray());
        }

        public TensorShape(params int[] dims) : base(dims)
        {

        }

        public TensorShape this[Slice slice]
        {
            get
            {
                return new TensorShape(Dimensions.Skip(slice.Start.Value)
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
            return Dimensions != null && Dimensions.Count(x => x < 1) == 0;
        }

        public bool is_compatible_with(TensorShape shape2)
        {
            throw new NotImplementedException("TensorShape is_compatible_with");
        }

        public static implicit operator TensorShape(int[] dims) => new TensorShape(dims);
        public static implicit operator TensorShape((int, int) dims) => new TensorShape(dims.Item1, dims.Item2);
        public static implicit operator TensorShape((int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3);
        public static implicit operator TensorShape((int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);
    }
}
