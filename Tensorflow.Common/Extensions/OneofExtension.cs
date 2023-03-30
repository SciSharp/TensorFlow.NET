using OneOf;
using System;

namespace Tensorflow.Common.Extensions
{
    public static class OneofExtension
    {
        public static bool IsTypeOrDeriveFrom<T>(this IOneOf src)
        {
            return src.Value is T;
        }
    }
}
