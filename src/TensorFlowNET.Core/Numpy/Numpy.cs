/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

namespace Tensorflow.NumPy;

public partial class np
{
    /// <summary>
    ///     A convenient alias for None, useful for indexing arrays.
    /// </summary>
    /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.indexing.html<br></br><br></br>https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number</remarks>
    public static readonly Slice newaxis = new Slice(null, null, 1) { IsNewAxis = true };

    // https://docs.scipy.org/doc/numpy-1.16.0/user/basics.types.html
    #region data type
    public static readonly TF_DataType @bool = TF_DataType.TF_BOOL;
    public static readonly TF_DataType @char = TF_DataType.TF_INT8;
    public static readonly TF_DataType @byte = TF_DataType.TF_INT8;
    public static readonly TF_DataType uint8 = TF_DataType.TF_UINT8;
    public static readonly TF_DataType ubyte = TF_DataType.TF_UINT8;
    public static readonly TF_DataType int16 = TF_DataType.TF_INT16;
    public static readonly TF_DataType uint16 = TF_DataType.TF_UINT16;
    public static readonly TF_DataType int32 = TF_DataType.TF_INT32;
    public static readonly TF_DataType uint32 = TF_DataType.TF_UINT32;
    public static readonly TF_DataType int64 = TF_DataType.TF_INT64;
    public static readonly TF_DataType uint64 = TF_DataType.TF_UINT64;
    public static readonly TF_DataType float32 = TF_DataType.TF_FLOAT;
    public static readonly TF_DataType float64 = TF_DataType.TF_DOUBLE;
    public static readonly TF_DataType @double = TF_DataType.TF_DOUBLE;
    public static readonly TF_DataType @decimal = TF_DataType.TF_DOUBLE;
    public static readonly TF_DataType complex_ = TF_DataType.TF_COMPLEX;
    public static readonly TF_DataType complex64 = TF_DataType.TF_COMPLEX64;
    public static readonly TF_DataType complex128 = TF_DataType.TF_COMPLEX128;
    public static readonly TF_DataType @string = TF_DataType.TF_STRING;
    public static readonly TF_DataType @object = TF_DataType.TF_VARIANT;
    #endregion

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
        => a.Equals(b);

    public static bool allclose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8,
        bool equal_nan = false) => throw new NotImplementedException("");

    public static RandomizedImpl random = new RandomizedImpl();
    public static LinearAlgebraImpl linalg = new LinearAlgebraImpl();
}
