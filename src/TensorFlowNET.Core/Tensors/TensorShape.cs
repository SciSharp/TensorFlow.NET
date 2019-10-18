using Newtonsoft.Json;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    ///     Represents the shape of a `Tensor`.
    /// </summary>
    /// <remarks>https://www.tensorflow.org/api_docs/python/tf/TensorShape</remarks>
    public class TensorShape
    {
        private readonly Shape shape;

        /// <summary>
        ///     Returns a list of Dimensions, or None if the shape is unspecified.
        /// </summary>
        public int[] dims => shape.Dimensions;

        /// <summary>
        ///     Returns the rank of this shape.
        /// </summary>
        public int ndim => rank;

        private int _rank;
        /// <summary>
        ///     Returns the rank of this shape.
        /// </summary>
        public int rank => _rank > -1 ? shape.NDim : -1;

        /// <summary>
        ///     Returns the size this shape represents.
        /// </summary>
#if SERIALIZABLE
        [JsonIgnore]
#endif
        public int size
        {
            get
            {
                var dims = shape.Dimensions;
                var computed = 1;
                for (int i = 0; i < dims.Length; i++)
                {
                    var val = dims[i];
                    if (val <= 0)
                        continue;
                    computed *= val;
                }

                return computed;
            }
        }

        public TensorShape()
        {
            _rank = -1;
            shape = new Shape();
        }

        public TensorShape(TensorShapeProto proto)
        {
            if (proto.UnknownRank) return;
            switch (proto.Dim.Count)
            {
                case 0: shape = new Shape(new int[0]); break;
                case 1: shape = Shape.Vector((int) proto.Dim[0].Size); break;
                case 2: shape = Shape.Matrix((int) proto.Dim[0].Size, (int) proto.Dim[1].Size); break;
                default:
                    var protodims = proto.Dim;
                    var len = protodims.Count;
                    var dims = new int[len];
                    for (int i = 0; i < len; i++) 
                        dims[i] = (int) protodims[i].Size;


                    shape = new Shape(dims); break;
            }
        }

        public TensorShape(params int[] dims)
        {
            switch (dims.Length)
            {
                case 0: shape = new Shape(new int[0]); break;
                case 1: shape = Shape.Vector(dims[0]); break;
                case 2: shape = Shape.Matrix(dims[0], dims[1]); break;
                default: shape = new Shape(dims); break;
            }
        }

        public TensorShape(int[][] dims)
        {
            if(dims.Length == 1)
            {
                switch (dims[0].Length)
                {
                    case 0: shape = new Shape(new int[0]); break;
                    case 1: shape = Shape.Vector((int)dims[0][0]); break;
                    case 2: shape = Shape.Matrix(dims[0][0], dims[1][2]); break;
                    default: shape = new Shape(dims[0]); break;
                }
            }
            else
            {
                throw new NotImplementedException("TensorShape int[][] dims");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="slice"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">When <see cref="Slice"/> is not an Index.</exception>
        [SuppressMessage("ReSharper", "PossibleInvalidOperationException")]
        public TensorShape this[Slice slice]
        {
            get
            {
                if (slice.Start.HasValue == false || slice.Length.HasValue == false)
                    throw new ArgumentException("Slice must has Start and Length.");

                return new TensorShape(dims.Skip(slice.Start.Value)
                    .Take(slice.Length.Value)
                    .ToArray());
            }
        }

        /// <summary>
        ///     Returns True iff `self` is fully defined in every dimension.
        /// </summary>
        /// <returns></returns>
        public bool is_fully_defined()
        {
            return rank > -1 && dims != null && dims.Count(x => x < 1) == 0;
        }

        public bool is_compatible_with(TensorShape shape2)
        {
            throw new NotImplementedException("TensorShape is_compatible_with");
        }

        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public TensorShape with_rank_at_least(int rank)
        {
            if (rank != ndim)
                throw new ValueError($"Shape {this} must have rank at least {rank}");
            else
                return this;
        }

        public TensorShape with_rank(int rank)
        {
            return merge_with(unknown_shape(rank: rank));
        }

        /// <summary>
        /// Returns an unknown TensorShape, optionally with a known rank.
        /// </summary>
        /// <param name="rank"></param>
        /// <returns></returns>
        public TensorShape unknown_shape(int rank = -1)
        {
            if (rank == -1)
                return new TensorShape(-1);
            else
                return new TensorShape(Enumerable.Repeat(-1, rank).ToArray());
        }

        /// <summary>
        ///     Returns the concatenation of the dimension in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorShape concatenate(int[] other)
        {
            return concatenate(new TensorShape(other));
        }

        /// <summary>
        ///     Returns the concatenation of the dimension in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TensorShape concatenate(TensorShape other)
        {
            var otherShape = other;

            if (ndim < 0 || otherShape.ndim < 0)
                return new TensorShape();
            else
            {
                var concatenate_dims = new int[ndim + otherShape.ndim];
                for (int i = 0; i < ndim; i++)
                    concatenate_dims[i] = dims[i];

                for (int i = 0; i < otherShape.ndim; i++)
                    concatenate_dims[ndim + i] = otherShape.dims[i];

                return new TensorShape(concatenate_dims);
            }
        }

        /// <summary>
        /// Returns a `TensorShape` combining the information in `self` and `other`.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TensorShape merge_with(TensorShape other)
        {
            if (dims == null)
                return other;

            var new_dims = new List<int>();

            foreach (var i in range(ndim))
            {
                var dim = new Dimension(dims[i]);
                var merged = dim.merge_with(new Dimension(other.dims[i]));
                new_dims.Add(merged.value);
            }

            return new TensorShape(new_dims.ToArray());
        }

        /// <summary>
        ///     Returns a cloned array from <see cref="dims"/>.
        /// </summary>
        public int[] as_list() {
            if (shape.IsEmpty)
                throw new ValueError("as_list() is not defined on an unknown TensorShape.");
            return (int[]) dims.Clone();
        }

        public override string ToString()
        {
            return shape.ToString();
        }

        public static implicit operator TensorShape(Shape shape) => new TensorShape((int[]) shape.Dimensions.Clone());
        public static implicit operator Shape(TensorShape shape) => new Shape((int[]) shape.dims.Clone());
        
        public static implicit operator int[](TensorShape shape) => (int[])shape.dims.Clone(); //we clone to avoid any changes
        public static implicit operator TensorShape(int[] dims) => new TensorShape(dims);

        public static explicit operator int(TensorShape shape) => shape.size;
        public static implicit operator TensorShape(int dim) => new TensorShape(dim);

        public static explicit operator (int, int)(TensorShape shape) => shape.dims.Length == 2 ? (shape.dims[0], shape.dims[1]) : (0, 0);
        public static implicit operator TensorShape((int, int) dims) => new TensorShape(dims.Item1, dims.Item2);

        public static explicit operator (int, int, int)(TensorShape shape) => shape.dims.Length == 3 ? (shape.dims[0], shape.dims[1], shape.dims[2]) : (0, 0, 0);
        public static implicit operator TensorShape((int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3);

        public static explicit operator (int, int, int, int)(TensorShape shape) => shape.dims.Length == 4 ? (shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]) : (0, 0, 0, 0);
        public static implicit operator TensorShape((int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static explicit operator (int, int, int, int, int)(TensorShape shape) => shape.dims.Length == 5 ? (shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3], shape.dims[4]) : (0, 0, 0, 0, 0);
        public static implicit operator TensorShape((int, int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static explicit operator (int, int, int, int, int, int)(TensorShape shape) => shape.dims.Length == 6 ? (shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3], shape.dims[4], shape.dims[5]) : (0, 0, 0, 0, 0, 0);
        public static implicit operator TensorShape((int, int, int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);
                
        public static explicit operator (int, int, int, int, int, int, int)(TensorShape shape) => shape.dims.Length == 7 ? (shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3], shape.dims[4], shape.dims[5], shape.dims[6]) : (0, 0, 0, 0, 0, 0, 0);
        public static implicit operator TensorShape((int, int, int, int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6, dims.Item7);
                
        public static explicit operator (int, int, int, int, int, int, int, int)(TensorShape shape) => shape.dims.Length == 8 ? (shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3], shape.dims[4], shape.dims[5], shape.dims[6], shape.dims[7]) : (0, 0, 0, 0, 0, 0, 0, 0);
        public static implicit operator TensorShape((int, int, int, int, int, int, int, int) dims) => new TensorShape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6, dims.Item7, dims.Item8);
    }
}
