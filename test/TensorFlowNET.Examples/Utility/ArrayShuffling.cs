using System;

namespace TensorFlowNET.Examples.Utility
{
    public static class ArrayShuffling
    {
        public static T[] Shuffle<T>(this Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
            return array;
        }
    }
}
