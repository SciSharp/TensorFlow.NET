using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public record Axis(params int[] axis)
    {
        public int this[int index] => axis[index];

        public static implicit operator int[]?(Axis axis)
            => axis?.axis;

        public static implicit operator Axis(int axis)
            => new Axis(axis);

        public static implicit operator Axis((int, int) axis)
            => new Axis(axis);

        public static implicit operator Axis((int, int, int) axis)
            => new Axis(axis);

        public static implicit operator Axis(int[] axis)
            => new Axis(axis);

        public static implicit operator Axis(long[] shape)
            => new Axis(shape.Select(x => (int)x).ToArray());

        public static implicit operator Axis(Shape shape)
            => new Axis(shape.dims.Select(x => (int)x).ToArray());
    }
}

namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}
