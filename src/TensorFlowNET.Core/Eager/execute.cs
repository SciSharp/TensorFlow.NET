using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using Tensorflow.Contexts;
using static Tensorflow.ApiDef.Types;
using static Tensorflow.CostGraphDef.Types;
using static Tensorflow.Binding;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    internal static class _execute
    {
        public static (DataType[], Tensor[]) onvert_to_mixed_eager_tensors(Tensor[] values, Context ctx)
        {
            var v = values.Select(t => ops.convert_to_tensor(t, ctx:ctx));
            var types = v.Select(t => t.dtype.as_datatype_enum());
            return (types.ToArray(), v.ToArray());
        }
        public static Tensor[] execute(string op_name, int num_outputs, Tensor[] inputs, object[] attrs, Context ctx, string name = null)
        {
            return quick_execute(op_name, num_outputs, inputs, attrs, ctx, name);
        }
        public static Tensor[] quick_execute(string op_name, int num_outputs, Tensor[] inputs, object[] attrs, Context ctx, string name = null)
        {
            string device_name = ctx.DeviceName;

            ctx.ensure_initialized();
            var tensors = tf.Runner.TFE_Execute(ctx, device_name, op_name, inputs, attrs, num_outputs);

            return tensors;
        }
        public static bool must_record_gradient()
        {
            return tf.GetTapeSet().Count != 0;
        }

        public static bool record_gradient(string op_name, Tensor[] inputs, object[] attrs, Tensor[] results)
        {
            return tf.Runner.RecordGradient(op_name, inputs, attrs, results);
        }
    }
}
