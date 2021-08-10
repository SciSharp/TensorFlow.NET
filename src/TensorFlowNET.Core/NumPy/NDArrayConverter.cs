using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.NumPy
{
    public class NDArrayConverter
    {
        public unsafe static T Scalar<T>(NDArray nd) where T : unmanaged
            => nd.dtype switch
            {
                TF_DataType.TF_FLOAT => Scalar<T>(*(float*)nd.data),
                TF_DataType.TF_INT64 => Scalar<T>(*(long*)nd.data),
                _ => throw new NotImplementedException("")
            };

        static T Scalar<T>(float input)
            => Type.GetTypeCode(typeof(T)) switch
            {
                TypeCode.Int32 => (T)Convert.ChangeType(input, TypeCode.Int32),
                TypeCode.Single => (T)Convert.ChangeType(input, TypeCode.Single),
                _ => throw new NotImplementedException("")
            };

        static T Scalar<T>(long input)
            => Type.GetTypeCode(typeof(T)) switch
            {
                TypeCode.Int32 => (T)Convert.ChangeType(input, TypeCode.Int32),
                TypeCode.Single => (T)Convert.ChangeType(input, TypeCode.Single),
                _ => throw new NotImplementedException("")
            };
    }
}
