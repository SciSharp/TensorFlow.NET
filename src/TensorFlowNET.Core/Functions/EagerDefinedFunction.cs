using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Contexts;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    public class EagerDefinedFunction
    {
        public int _num_outputs;
        FuncGraph _func_graph;
        FunctionDef _definition;
        public string Name => _func_graph.FuncName;
        public FunctionDef Definition
        {
            get
            {
                if(_definition is null)
                {
                    _definition = _get_definition();
                }
                return _definition;
            }
        }
        public EagerDefinedFunction(string name, FuncGraph graph, 
            Tensors inputs, Tensors outputs,
            Dictionary<string, string> attrs)
        {
            _num_outputs = outputs.Length;
            
            var input_ops = inputs.Select(x => x.op).ToArray();
            var operations = graph.get_operations().Where(x => !input_ops.Contains(x.op))
                .Select(x => x as Operation).ToArray();
            var output_names = new string[0];

            _func_graph = new FuncGraph(graph, name, attrs);
            _func_graph.ToGraph(operations, inputs, outputs, output_names);
        }

        public Tensors Call(Tensors args)
        {
            var attrs = new object[]
                {
                    "executor_type", "",
                    "config_proto", tf.Context.FunctionCallOptions.config_proto_serialized()
                };

            var results = tf.Runner.TFE_Execute(tf.Context,
                tf.Context.DeviceName,
                _func_graph.FuncName,
                args,
                attrs,
                _num_outputs);

            return results;
        }

        public void AddToGraph(Graph g = null)
        {
            if(g is null && tf.Context.executing_eagerly())
            {
                var ctx = tf.Context;
                if (!ctx.has_function(this.Name))
                {
                    ctx.add_function_def(Definition);
                }
            }
            else
            {
                if (!g.IsFunction(Name))
                {
                    g.AddFunction(this);
                }
                foreach(var f in _func_graph.Functions.Values)
                {
                    if (!g.IsFunction(f.Name))
                    {
                        g.AddFunction(f);
                    }
                }
            }
        }

        private FunctionDef _get_definition()
        {
            var buffer = c_api_util.tf_buffer();
            // TODO(Rinne): pywrap_tf_session.TF_FunctionToFunctionDef
            var proto_data = c_api.TF_GetBuffer(buffer);
            throw new NotImplementedException();
        }
    }
}
