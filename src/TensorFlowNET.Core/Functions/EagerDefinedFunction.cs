using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    public class EagerDefinedFunction
    {
        public int _num_outputs;
        public string Name => _func_graph.FuncName;

        FuncGraph _func_graph;
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
    }
}
