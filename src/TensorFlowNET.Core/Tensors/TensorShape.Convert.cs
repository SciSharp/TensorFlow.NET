using NumSharp;

namespace Tensorflow
{
    public partial class TensorShape
    {
        public void Deconstruct(out int h, out int w)
        {
            h = dims[0];
            w = dims[1];
        }

        public static implicit operator TensorShape(Shape shape) => new TensorShape((int[])shape.Dimensions.Clone());
        public static implicit operator Shape(TensorShape shape) => new Shape((int[])shape.dims.Clone());

        public static implicit operator int[](TensorShape shape) => shape == null ? null : (int[])shape.dims.Clone(); //we clone to avoid any changes
        public static implicit operator TensorShape(int[] dims) => dims == null ? null : new TensorShape(dims);

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
