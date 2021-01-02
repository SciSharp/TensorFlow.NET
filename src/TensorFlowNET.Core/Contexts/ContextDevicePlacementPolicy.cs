using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Contexts
{
    public enum ContextDevicePlacementPolicy
    {
        // Running operations with input tensors on the wrong device will fail.
        DEVICE_PLACEMENT_EXPLICIT = 0,
        // Copy the tensor to the right device but log a warning.
        DEVICE_PLACEMENT_WARN = 1,
        // Silently copy the tensor, which has a performance cost since the operation
        // will be blocked till the copy completes. This is the default placement
        // policy.
        DEVICE_PLACEMENT_SILENT = 2,
        // Placement policy which silently copies int32 tensors but not other dtypes.
        DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3,
    }
}
