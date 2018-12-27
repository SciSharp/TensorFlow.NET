using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using TF_DataType = Tensorflow.DataType;
using attr_value_pb2 = Tensorflow;
using Tensorflow.Eager;

namespace Tensorflow
{
    public static partial class tf
    {
        public static TF_DataType float32 = TF_DataType.TF_FLOAT;
        public static TF_DataType chars = TF_DataType.TF_STRING;

        public static Context context = new Context();

        public static Graph g = new Graph(c_api.TF_NewGraph());

        public static object Variable<T>(T data, TF_DataType dtype)
        {
            return new Variable(null, TF_DataType.DtInvalid);
        }

        public static unsafe Tensor add(Tensor a, Tensor b)
        {
            return gen_math_ops.add(a, b);
        }

        public static unsafe Tensor placeholder(TF_DataType dtype, TensorShape shape = null)
        {
            return gen_array_ops.placeholder(dtype, shape);
        }

        public static void enable_eager_execution()
        {
            context.default_execution_mode = Context.EAGER_MODE;
        }

        public static string VERSION => Marshal.PtrToStringAnsi(c_api.TF_Version());

        public static Graph get_default_graph()
        {
            return ops.get_default_graph();
        }

        public static Graph Graph()
        {
            return g;
        }

        public static Session Session()
        {
            return new Session();
        }
    }
}
