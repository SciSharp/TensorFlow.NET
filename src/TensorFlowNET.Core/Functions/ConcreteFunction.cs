using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Framework.Models;
using Tensorflow.Gradients;
using Tensorflow.Graphs;
using Tensorflow.Train;
using Tensorflow.Util;
using Tensorflow.Common.Extensions;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    /// <summary>
    /// 
    /// </summary>
    public class ConcreteFunction: Trackable
    {
        protected IEnumerable<Tensor> _captured_inputs;
        protected DelayedRewriteGradientFunctions _delayed_rewrite_functions;
        protected Dictionary<string, AttrValue> _attrs;
        protected FunctionSpec _function_spec;
        protected FunctionSpec _pre_initialized_function_spec = null;
        protected EagerDefinedFunction _inference_function;
        protected Dictionary<string, TapeGradientFunctions> _tape_functions_cache = new();
        internal FuncGraph func_graph;
        internal ForwardBackwardCall forward_backward;
        public Tensor[] Inputs => func_graph.Inputs;
        public Tensor[] CapturedInputs => func_graph.external_captures;

        public string Name => _delayed_rewrite_functions.Forward().Name;

        public Tensor[] Outputs => func_graph.Outputs;
        public Type ReturnType;
        public TensorSpec[] OutputStructure;
        public IEnumerable<string> ArgKeywords { get; set; }
        public long NumPositionArgs { get; set; }
        public FunctionDef FunctionDef => _delayed_rewrite_functions.Forward().Definition;
        public Tensor[] FlatStructuredOutputs => func_graph.FlatStructuredOutputs;
        public IEnumerable<IVariableV1> Variables => func_graph.Variables;
        public IEnumerable<IVariableV1> TrainableVariables => func_graph.TrainableVariables;
        internal NameAttrList AsNameAttrList
        {
            get
            {
                NameAttrList ret = new() { Name = this.Name };
                foreach (var (name, value) in _attrs)
                {
                    ret.Attr[name] = value;
                }
                return ret;
            }
        }

        public ConcreteFunction(string name)
        {
            func_graph = new FuncGraph(name);
            _captured_inputs = func_graph.external_captures;
            _attrs= new Dictionary<string, AttrValue>();
            _set_infer_function();
        }

        public ConcreteFunction(FuncGraph graph, Dictionary<string, AttrValue> attrs = null)
        {
            func_graph = graph;
            _captured_inputs = func_graph.external_captures;

            //ToGraph(graph.Inputs, graph.Outputs.Where(x => x != null).ToArray());
            _attrs = attrs;
            _set_infer_function();
        }

        public ConcreteFunction(Func<Tensor, Tensor> func, TF_DataType dtype)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            func_graph = new FuncGraph(func_name);
            func_graph.as_default();
            var input = tf.placeholder(dtype);
            var output = func(input);

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
                new[] { input },
                new[] { output },
                null);
            func_graph.Exit();
            _captured_inputs = func_graph.external_captures;
            _attrs = new Dictionary<string, AttrValue>();
            _set_infer_function();
        }

        public ConcreteFunction(Func<Tensor, IDatasetV2> func, TF_DataType dtype)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            func_graph = new FuncGraph(func_name);
            func_graph.as_default();

            var input = tf.placeholder(dtype);
            var output = func(input);

            OutputStructure = output.structure;

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
                new[] { input },
                new[] { output.variant_tensor },
                null);
            func_graph.Exit();
            _captured_inputs = func_graph.external_captures;
            _attrs = new Dictionary<string, AttrValue>();
            _set_infer_function();
        }

        /*public ConcreteFunction(Func<Tensors, Tensors> func,
            TF_DataType[] dtypes, Shape[] shapes)
        {
            string func_name = $"{func.Method.Name}_{ops.uid_function()}";

            // IntPtr func_handle;
            func_graph = new FuncGraph(func_name);
            func_graph.as_default();

            var inputs = new Tensors();
            foreach(var (i, dtype) in enumerate(dtypes))
                inputs.Add(tf.placeholder(dtypes[i], shape: shapes[i], name: "args"));
            Outputs = func(inputs);
            OutputStructure = Outputs.Select(x => x.ToTensorSpec()).ToArray();

            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers, inputs, Outputs, null);
            func_graph.Exit();
        }*/

        public void ToGraph(Tensors inputs, Tensors outputs)
        {
            var opers = func_graph._nodes_by_name.Values.Select(x => x as Operation).ToArray();
            func_graph.ToGraph(opers,
                inputs,
                outputs,
                null);
            OutputStructure = outputs.Select(x => x.ToTensorSpec()).ToArray();
        }

        public void Enter()
        {
            func_graph.as_default();
        }

        public void Exit()
        {
            func_graph.Exit();
        }

        public Tensors FilteredCall(Tensors inputs)
        {
            return CallFlat(inputs, CapturedInputs);
        }

        /// <summary>
        /// Executes the wrapped function.
        /// </summary>
        /// <param name="args"></param>
        /// <param name="captured_inputs"></param>
        /// <returns></returns>
        public Tensors CallFlat(Tensor[] args, Tensor[] captured_inputs)
        {
            var executing_eagerly = tf.Context.executing_eagerly();
            var default_graph = ops.get_default_graph();
            // TODO(Rinne): deal with `default_graph.building_function`

            var tempvv = func_graph.Variables;
            if(tf.GetTapeSet().Count > 0 || default_graph is FuncGraph)
            {
                foreach(var v in this.func_graph.Variables)
                {
                    resource_variable_ops.variable_accessed(v);
                }
            }

            var tensor_inputs = new Tensors();
            foreach (var (i, arg) in enumerate(args))
            {
                tensor_inputs.Add(arg);
                // If we're graph building, shape inference is on.
            }
            if (!executing_eagerly)
            {
                // TODO(Rinne): add the check
            }
            tensor_inputs.AddRange(captured_inputs); 

            args = tensor_inputs.ToArray();

            var possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args);
            // No tape is watching; skip to running the function.
            if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE && executing_eagerly)
            {
                return _build_call_outputs(_inference_function.Call(args));
            }

            forward_backward = SelectForwardAndBackwardFunctions(args, possible_gradient_type, executing_eagerly);
            var (forward_function, args_with_tangents) = forward_backward.Forward();
            Tensors flat_outputs = null;
            if (executing_eagerly)
            {
                flat_outputs = forward_function.Call(args_with_tangents);
            }
            else
            {
                tf_with(default_graph._override_gradient_function(new Dictionary<string, Func<Operation, object[], Tensor[]>>(){
                    { "PartitionedCall", _get_gradient_function() }, { "StatefulPartitionedCall", _get_gradient_function() }
                }), _ =>
                {
                    flat_outputs = forward_function.Call(args_with_tangents);
                });
            }
            forward_backward.Record(flat_outputs);
            return _build_call_outputs(flat_outputs);
        }

        public void AddTograph(Graph? g = null)
        {
            if(!tf.Context.executing_eagerly() && g is null)
            {
                g = ops.get_default_graph();
            }
            _delayed_rewrite_functions.Forward().AddToGraph(g);
        }

        public void SetExternalCaptures(IEnumerable<Tensor> captures)
        {
            _captured_inputs = captures;
        }

        ForwardBackwardCall SelectForwardAndBackwardFunctions(Tensors args, int possible_gradient_type, bool executing_eagerly)
        {
            TangentInfo input_tangents;
            if (executing_eagerly)
            {
                // TODO(Rinne): check if it needs to be implemented.
                input_tangents = new TangentInfo();
            }
            else
            {
                input_tangents = new TangentInfo();
            }
            if(possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER)
            {
                if(input_tangents.Indices is not null || executing_eagerly)
                {
                    string cache_key = "first_order";
                    if(!_tape_functions_cache.TryGetValue(cache_key, out var functions))
                    {
                        functions = new FirstOrderTapeGradientFunctions(func_graph, false);
                        _tape_functions_cache[cache_key] = functions;
                    }
                    return new ForwardBackwardCall(functions, args, tape_watching: true);
                }
                else
                {
                    return new ForwardBackwardCall(_delayed_rewrite_functions, args, tape_watching: true);
                }
            }
            else if(possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER)
            {
                throw new NotImplementedException();
            }

            // TODO(Rinne): add arg "input_tagents" for ForwardBackwardCall.
            return new ForwardBackwardCall(_delayed_rewrite_functions, args, tape_watching: false);
        }

        internal void set_variables(IEnumerable<IVariableV1> variables)
        {
            func_graph.Variables = variables;
        }

        internal void _set_infer_function()
        {
            _delayed_rewrite_functions = new DelayedRewriteGradientFunctions(func_graph, _attrs);
            _inference_function = _delayed_rewrite_functions.Forward();
        }

        internal void _set_function_spec(FunctionSpec spec)
        {
            _function_spec = null;
            _pre_initialized_function_spec = spec;
            _initialize_function_spec();
        }

        internal void _initialize_function_spec()
        {
            if(_pre_initialized_function_spec is null)
            {
                return;
            }
            Debug.Assert(_function_spec is null, "already initialized");
            var spec = _pre_initialized_function_spec;
            //var args = spec.Fullargspec.DictValue.Fields["args"];
            // TODO(Rinne): self.structured_input_signature

            _function_spec = new FunctionSpec()
            {
                Fullargspec = spec.Fullargspec,
                IsMethod = spec.IsMethod,
                InputSignature = spec.InputSignature
            };
        }

        internal Func<Operation, object[], Tensor[]> _get_gradient_function()
        {
            return _delayed_rewrite_functions._rewrite_forward_and_call_backward;
        }

        private Tensors _build_call_outputs(Tensors result)
        {
            // TODO(Rinne): deal with `func_graph.structured_outputs`

            return result;
        }

        public override string ToString()
            => Name;
    }
}
