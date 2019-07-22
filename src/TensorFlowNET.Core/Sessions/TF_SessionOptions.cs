using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_SessionOptions
    {
        public SessionOptions options;
    }
}
