using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using TF_DataType = Tensorflow.DataType;

namespace TensorFlowNET.Core
{
    public static class Tensorflow
    {
        public delegate void Deallocator(IntPtr data, IntPtr size, IntPtr deallocatorData);

        public static unsafe Tensor constant(object value)
        {
            var g = ops.get_default_graph();
            g.create_op("Const", value, new TF_DataType[] { TF_DataType.DtDouble });

            return new Tensor();
        }

        public static Deallocator FreeTensorDataDelegate = FreeTensorData;

        [MonoPInvokeCallback(typeof(Deallocator))]
        internal static void FreeTensorData(IntPtr data, IntPtr len, IntPtr closure)
        {
            Marshal.FreeHGlobal(data);
        }

        public static string VERSION => Marshal.PtrToStringAnsi(c_api.TF_Version());

        public static Graph get_default_graph()
        {
            return ops.get_default_graph();
        }

        public static Graph Graph()
        {
            Graph g = new Graph(c_api.TF_NewGraph());
            return g;
        }
    }
}
