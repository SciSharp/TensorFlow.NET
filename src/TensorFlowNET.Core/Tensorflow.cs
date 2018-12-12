using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorFlowNET.Core
{
    public static class Tensorflow
    {
        public const string TensorFlowLibName = "libtensorflow";

        [DllImport(TensorFlowLibName)]
        public static extern unsafe IntPtr TF_Version();

        public static string VERSION => Marshal.PtrToStringAnsi(TF_Version());

        [DllImport(TensorFlowLibName)]
        static extern unsafe IntPtr TF_NewOperation(IntPtr graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        static extern unsafe IntPtr TF_FinishOperation(IntPtr desc, IntPtr status);

        public static IntPtr constant<T>(T value)
        {
            var g = Graph();
            return TF_NewOperation(g.TFGraph, "Const", "Const");
        }

        [DllImport(TensorFlowLibName)]
        static extern unsafe IntPtr TF_NewGraph();

        public static Graph Graph()
        {
            Graph g = new Graph();
            g.TFGraph = TF_NewGraph();
            return g;
        }
    }
}
