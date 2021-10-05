using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Lite
{
    public enum TfLiteStatus
    {
        kTfLiteOk = 0,

        // Generally referring to an error in the runtime (i.e. interpreter)
        kTfLiteError = 1,

        // Generally referring to an error from a TfLiteDelegate itself.
        kTfLiteDelegateError = 2,

        // Generally referring to an error in applying a delegate due to
        // incompatibility between runtime and delegate, e.g., this error is returned
        // when trying to apply a TfLite delegate onto a model graph that's already
        // immutable.
        kTfLiteApplicationError = 3,

        // Generally referring to serialized delegate data not being found.
        // See tflite::delegates::Serialization.
        kTfLiteDelegateDataNotFound = 4,

        // Generally referring to data-writing issues in delegate serialization.
        // See tflite::delegates::Serialization.
        kTfLiteDelegateDataWriteError = 5,
    }
}
