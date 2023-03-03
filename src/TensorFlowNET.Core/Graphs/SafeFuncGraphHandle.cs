using Tensorflow.Util;

namespace Tensorflow;

public sealed class SafeFuncGraphHandle : SafeTensorflowHandle
{
    private SafeFuncGraphHandle()
    {
    }

    public SafeFuncGraphHandle(IntPtr handle)
        : base(handle)
    {
    }

    protected override bool ReleaseHandle()
    {
        c_api.TF_DeleteFunction(handle);
        SetHandle(IntPtr.Zero);
        return true;
    }
}
