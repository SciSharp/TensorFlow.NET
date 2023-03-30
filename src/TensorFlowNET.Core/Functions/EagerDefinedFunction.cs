using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Contexts;
using Tensorflow.Graphs;
using Tensorflow.Operations;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    public class EagerDefinedFunction
    {
        public int _num_outputs;
        FuncGraph _func_graph;
        FunctionDef _definition;
        Tensor[] _func_graph_outputs;
        public string Name => _func_graph.FuncName;
        public DataType[] OutputTypes { get; protected set; }
        public Shape[] OutputShapes { get; protected set; }
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
            OutputShapes = outputs.Select(x => x.shape).ToArray();
            OutputTypes = outputs.Select(x => x.dtype.as_datatype_enum()).ToArray();

            _func_graph = new FuncGraph(graph, name, attrs);
            _func_graph_outputs = new List<Tensor>(outputs).ToArray();
            _func_graph.ToGraph(operations, inputs, outputs, output_names);
        }

        public Tensors Call(Tensors args)
        {
            // TODO(Rinne): Add arg `CancellationManager`.
            // TODO(Rinne): Check the arg length.
            var function_call_options = tf.Context.FunctionCallOptions;
            string config;
            if (string.IsNullOrEmpty(function_call_options.config_proto_serialized()))
            {
                config = function_utils.get_disabled_rewriter_config();
            }
            else
            {
                config = function_call_options.config_proto_serialized();
            }
            // TODO(Rinne): executor_type
            var executing_eagerly = tf.Context.executing_eagerly();

            var attrs = new object[]
                {
                    "executor_type", "",
                    "config_proto", tf.Context.FunctionCallOptions.config_proto_serialized()
                };

            Tensor[] outputs;
            if (executing_eagerly)
            {
                outputs = tf.Runner.TFE_Execute(tf.Context,
                tf.Context.DeviceName,
                _func_graph.FuncName,
                args,
                attrs,
                _num_outputs);
            }
            else
            {
                tf.GradientTape().stop_recording();
                outputs = functional_ops.partitioned_call(args, this, OutputTypes,
                    executing_eagerly, config, "");
            }
            foreach(var (i, func_graph_output) in enumerate(_func_graph_outputs))
            {
                handle_data_util.copy_handle_data(func_graph_output, outputs[i]);
            }
            if (executing_eagerly)
            {
                return outputs;
            }
            else
            {
                foreach(var (i, shape) in enumerate(OutputShapes))
                {
                    outputs[i].shape = shape;
                }
                return outputs;
            }
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
