using System.Runtime.InteropServices;

namespace Tensorflow.Functions
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Function
    {
        FunctionDef fdef;
    }
}
