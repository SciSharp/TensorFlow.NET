using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Util
{
    public static class Arrays
    {
        public static Type ResolveElementType(this Array arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            var t = arr.GetType().GetElementType();
            // ReSharper disable once PossibleNullReferenceException
            while (t.IsArray)
                t = t.GetElementType();

            return t;
        }
    }
}
