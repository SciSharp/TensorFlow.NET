using System;
using System.Collections.Generic;
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
    }
}

namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}
