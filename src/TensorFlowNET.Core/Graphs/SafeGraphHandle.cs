using Tensorflow.Util;

namespace Tensorflow;

public sealed class SafeGraphHandle : SafeTensorflowHandle
{
    private SafeGraphHandle()
    {
    }

    public SafeGraphHandle(IntPtr handle)
        : base(handle)
    {
    }

    protected override bool ReleaseHandle()
    {
        c_api.TF_DeleteGraph(handle);
        SetHandle(IntPtr.Zero);
        return true;
    }
}
