using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    public class gen_functional_ops
    {
        public static Tensor[] partitioned_call(Tensors args, TF_DataType[] tout, EagerDefinedFunction f, 
            string config = "", string config_proto = "", string executor_type = "", string name = null)
        {
            var ctx = tf.Context;
            if (ctx.executing_eagerly())
            {
                try
                {
                    return tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo("PartitionedCall", name,
                        args, tout, f, config, config_proto, executor_type));
                }
                catch (Exception)
                {

                }
            }

            if (config is null)
            {
                config = "";
            }
            if (config_proto is null)
            {
                config_proto = "";
            }
            if (executor_type is null)
            {
                executor_type = "";
            }
            Dictionary<string, object> kwargs = new();
            kwargs["args"] = args;
            kwargs["Tout"] = tout;
            kwargs["f"] = f;
            kwargs["config"] = config;
            kwargs["config_proto"] = config_proto;
            kwargs["executor_type"] = executor_type;
            var output = tf.OpDefLib._apply_op_helper("PartitionedCall",
                name, kwargs);
            var result = output.outputs;
            if (execute.must_record_gradient())
            {
                throw new NotImplementedException();
            }
            return result;
        }

        public static Tensor[] partitioned_call_eager_fallback(Tensors args, TF_DataType[] tout, EagerDefinedFunction f,
            string config, string config_proto, string executor_type, string name, Context ctx)
        {
            // TODO(Rinne): implement it.
            throw new NotImplementedException();
            if(config is null)
            {
                config = "";
            }
            if(config_proto is null)
            {
                config_proto = "";
            }
            if(executor_type is null)
            {
                executor_type = "";
            }
            object[] attrs = new object[]
            {

            };
        }
    }
}
