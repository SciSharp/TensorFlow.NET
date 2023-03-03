using Tensorflow.Util;

namespace Tensorflow.Checkpoint;

public sealed class SafeCheckpointReaderHandle : SafeTensorflowHandle
{
    private SafeCheckpointReaderHandle() : base ()
    {
    }

    public SafeCheckpointReaderHandle(IntPtr handle) : base(handle)
    {
    }

    protected override bool ReleaseHandle()
    {
        c_api.TF_DeleteCheckpointReader(handle);
        SetHandle(IntPtr.Zero);
        return true;
    }
}
