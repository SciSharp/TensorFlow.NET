using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Lite
{
    public enum TfLiteDataType
    {
        kTfLiteNoType = 0,
        kTfLiteFloat32 = 1,
        kTfLiteInt32 = 2,
        kTfLiteUInt8 = 3,
        kTfLiteInt64 = 4,
        kTfLiteString = 5,
        kTfLiteBool = 6,
        kTfLiteInt16 = 7,
        kTfLiteComplex64 = 8,
        kTfLiteInt8 = 9,
        kTfLiteFloat16 = 10,
        kTfLiteFloat64 = 11,
        kTfLiteComplex128 = 12,
        kTfLiteUInt64 = 13,
        kTfLiteResource = 14,
        kTfLiteVariant = 15,
        kTfLiteUInt32 = 16,
    }
}
