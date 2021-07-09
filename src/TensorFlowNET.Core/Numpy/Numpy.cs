using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        /// <summary>
        ///     A convenient alias for None, useful for indexing arrays.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.indexing.html<br></br><br></br>https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number</remarks>
        public static readonly Slice newaxis = new Slice(null, null, 1) { IsNewAxis = true };

        // https://docs.scipy.org/doc/numpy-1.16.0/user/basics.types.html
        public static readonly Type bool_ = typeof(bool);
        public static readonly Type bool8 = bool_;
        public static readonly Type @bool = bool_;

        public static readonly Type @char = typeof(char);

        public static readonly Type @byte = typeof(byte);
        public static readonly Type uint8 = typeof(byte);
        public static readonly Type ubyte = uint8;


        public static readonly Type int16 = typeof(short);

        public static readonly Type uint16 = typeof(ushort);

        public static readonly Type int32 = typeof(int);

        public static readonly Type uint32 = typeof(uint);

        public static readonly Type int_ = typeof(long);
        public static readonly Type int64 = int_;
        public static readonly Type intp = int_; //TODO! IntPtr?
        public static readonly Type int0 = int_;

        public static readonly Type uint64 = typeof(ulong);
        public static readonly Type uint0 = uint64;
        public static readonly Type @uint = uint64;

        public static readonly Type float32 = typeof(float);

        public static readonly Type float_ = typeof(double);
        public static readonly Type float64 = float_;
        public static readonly Type @double = float_;

        public static readonly Type complex_ = typeof(Complex);
        public static readonly Type complex128 = complex_;
        public static readonly Type complex64 = complex_;
        public static readonly Type @decimal = typeof(decimal);

        public static Type chars => throw new NotSupportedException("Please use char with extra dimension.");

        public static double nan => double.NaN;
        public static double NAN => double.NaN;
        public static double NaN => double.NaN;
        public static double pi => Math.PI;
        public static double e => Math.E;
        public static double euler_gamma => 0.57721566490153286060651209008240243d;
        public static double inf => double.PositiveInfinity;
        public static double infty => double.PositiveInfinity;
        public static double Inf => double.PositiveInfinity;
        public static double NINF => double.NegativeInfinity;
        public static double PINF => double.PositiveInfinity;
        public static double Infinity => double.PositiveInfinity;
        public static double infinity => double.PositiveInfinity;

        

        public static bool array_equal(NDArray a, NDArray b)
            => throw new NotImplementedException("");

        public static NDArray concatenate(NDArray[] arrays, int axis = 0) 
            => throw new NotImplementedException("");

        public static NDArray frombuffer(byte[] bytes, Type dtype) 
            => throw new NotImplementedException("");

        public static NDArray frombuffer(byte[] bytes, string dtype)
            => throw new NotImplementedException("");

        public static NDArray squeeze(NDArray x1)
            => throw new NotImplementedException("");

        public static bool allclose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8,
            bool equal_nan = false) => throw new NotImplementedException("");

        public static class random
        {
            public static NDArray permutation(int x)
            {
                throw new NotImplementedException("");
            }

            public static void shuffle(NDArray nd)
            {

            }

            public static NDArray rand(params int[] shape)
                => throw new NotImplementedException("");

            public static NDArray randint(long x)
                => throw new NotImplementedException("");

            public static NDArray RandomState(int x)
                => throw new NotImplementedException("");
        }

        public static NpzDictionary<T> Load_Npz<T>(byte[] bytes)
            where T : class, IList, ICloneable, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            throw new NotImplementedException("");
        }
    }
}
