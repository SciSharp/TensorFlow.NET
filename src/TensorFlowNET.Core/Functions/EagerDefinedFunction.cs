using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Graphs;
using Tensorflow.Operations;
using Tensorflow.Util;
using Tensorflow.Common.Extensions;
using static Tensorflow.Binding;
using Tensorflow.Framework;
using System.Buffers;
using Tensorflow.Gradients;

namespace Tensorflow.Functions
{
    public class EagerDefinedFunction: IDisposable
    {
        public int _num_outputs;
        FuncGraph _graph;
        FunctionDef _definition;
        OpDef _signature;
        string _name;
        internal ScopedTFFunction _c_func;
        internal Tensor[] _func_graph_outputs;
        internal string _grad_func_name;
        internal Func<Operation, Tensor[], Tensor[]> csharp_grad_func;
        internal EagerDefinedFunction _grad_func;
        internal bool _registered_on_context = false;
        public string Name => _name;
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

        public OpDef Signature
        {
            get
            {
                if( _signature is null)
                {
                    _signature = Definition.Signature;
                }
                return _signature;
            }
        }
        public unsafe EagerDefinedFunction(string name, FuncGraph graph, 
            Tensors inputs, Tensors outputs,
            Dictionary<string, AttrValue> attrs)
        {
            var input_ops = inputs.Select(x => x.op).ToArray();
            var operations = graph.get_operations().Where(x => !input_ops.Contains(x.op))
                .Select(x => x as Operation).ToArray();
            var graph_output_names = graph._output_names;
            string[] output_names;
            if(graph_output_names is not null && outputs.All(t => graph_output_names.ContainsKey(ops.tensor_id(t))))
            {
                output_names = outputs.Select(t => graph_output_names[ops.tensor_id(t)]).ToArray();
                if(output_names.Distinct().Count() != output_names.Length)
                {
                    output_names = new string[0];
                }
            }
            else
            {
                output_names = new string[0];
            }

            Status status = new Status();
            var fn = c_api.TF_GraphToFunction(graph.c_graph,
                name,
                false,
                operations.Length,
                operations.Length == 0 ? new IntPtr[0] : operations.Select(x => (IntPtr)x).ToArray(),
                inputs.Length,
                inputs.Select(t => t._as_tf_output()).ToArray(),
                outputs.Length,
                outputs.Select(t => t._as_tf_output()).ToArray(),
                output_names.Length != outputs.Length ? null : output_names, 
                IntPtr.Zero, // warning: the control output hasbben totally ignored.
                null,
                status);
            status.Check(true);

            _c_func = new ScopedTFFunction(fn, name);

            foreach(var (attr_name, attr_value) in attrs)
            {
                var serialized = attr_value.ToByteArray();
                c_api.TF_FunctionSetAttrValueProto(fn, attr_name, serialized, serialized.Length, status);
                status.Check(true);
            }

            var signature = _get_definition().Signature;
            _name = signature.Name;
            tf_with(ops.init_scope(), s =>
            {
                tf.Context.add_function(fn);
                _registered_on_context = true;
            });

            _num_outputs = signature.OutputArg.Count;
            OutputTypes = signature.OutputArg.Select(x => x.Type).ToArray();
            OutputShapes = outputs.Select(x => x.shape).ToArray();
            _func_graph_outputs = new List<Tensor>(outputs).ToArray();
            csharp_grad_func = null;
            _graph = graph;
        }

        public unsafe Tensors Call(Tensors args)
        {
            // TODO(Rinne): Add arg `CancellationManager`.
            // TODO(Rinne): Check the arg length.
            var function_call_options = tf.Context.FunctionCallOptions;
            string config = ""; // TODO(Rinne): revise it. The following code should work but not, for unclear reasons.

            //if (function_call_options.config_proto_serialized().Length == 0)
            //{
            //    config = function_utils.get_disabled_rewriter_config().ToStringUtf8();
            //}
            //else
            //{
            //    config = function_call_options.config_proto_serialized().ToStringUtf8();
            //}

            string executor_type = function_call_options.ExecutorType ?? "";
            var executing_eagerly = tf.Context.executing_eagerly();

            var attrs = new object[]
                {
                    "executor_type", executor_type,
                    "config_proto", config
                };

            Tensor[] outputs;
            if (executing_eagerly)
            {
                outputs = _execute.execute(
                    Signature.Name,
                    _num_outputs,
                    args,
                    attrs,
                    tf.Context);
            }
            else
            {
                if(tf.GetTapeSet().Count == 0)
                {
                    outputs = functional_ops.partitioned_call(args, this, OutputTypes,
                        executing_eagerly, config, "");
                }
                else
                {
                    var tape = tf.GetTapeSet().Peek();
                    tape.StopRecord();
                    outputs = functional_ops.partitioned_call(args, this, OutputTypes,
                        executing_eagerly, config, "");
                    tape.StartRecord();
                }
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
                foreach(var f in _graph.Functions.Values)
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
            Status status = new();
            c_api.TF_FunctionToFunctionDef(_c_func.Get(), buffer, status);
            status.Check(true);
            var proto_data = c_api.TF_GetBuffer(buffer);
            return FunctionDef.Parser.ParseFrom(proto_data.AsSpan<byte>());
        }

        public void Dispose()
        {
            tf.Context.remove_function(Name);
        }
    }
}
