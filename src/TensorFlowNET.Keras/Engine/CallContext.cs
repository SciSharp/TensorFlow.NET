using Keras.Layers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public class CallContext
    {
        public bool in_keras_graph
        {
            get
            {
                throw new NotImplementedException();
            }
        }
        public CallContext()
        {

        }

        public void enter(Layer layer, Tensor[] inputs, Graph build_graph, bool training) => throw new NotImplementedException();

        public bool training_arg_passed_to_call(string[] argspec, Dictionary<string, object> args, Dictionary<string, object>  kwargs) => throw new NotImplementedException();

        public dynamic autocast_context_manager(string dtype) => throw new NotImplementedException();

        public bool is_subclassed(Layer layer) => throw new NotImplementedException();

        public bool from_saved_model(Layer layer) => throw new NotImplementedException();

        public bool check_graph_consistency(Tensor tensor = null, string method = "add_loss", bool force_raise = false) => throw new NotImplementedException();

        public dynamic mark_as_return(Tensor[] outputs, dynamic acd) => throw new NotImplementedException();

        public MethodInfo Default(MemberInfo method) => throw new NotImplementedException();

        public void enable_v2_dtype_behavior() => throw new NotImplementedException();

        public void disable_v2_dtype_behavior() => throw new NotImplementedException();

        public void v2_dtype_behavior_enabled() => throw new NotImplementedException();
    }
}
