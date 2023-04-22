using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using Tensorflow.Framework;
using Tensorflow.Functions;
using Tensorflow.Gradients;
using Tensorflow.Graphs;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Training.Saving.SavedModel
{
    public static class function_deserialization
    {
        private static string _INFERENCE_PREFIX = "__inference_";
        private static string _FUNCTION_WRAPPER_NAME_REGEX = $@"^{_INFERENCE_PREFIX}(.*)_\d+$";
        /// <summary>
        /// Creates a `Function` from a `SavedFunction`.
        /// </summary>
        /// <param name="saved_concrete_function"></param>
        /// <param name="concrete_functions"></param>
        /// <returns></returns>
        public static Function recreate_function(SavedFunction saved_function,
            IDictionary<string, ConcreteFunction> concrete_functions)
        {
            var function_spec = _deserialize_function_spec_as_nonmethod(saved_function.FunctionSpec);

            Tensor[] restored_function_body(Tensor[] inputs)
            {
                if(saved_function.ConcreteFunctions is null || saved_function.ConcreteFunctions.Count == 0)
                {
                    throw new ValueError("Found zero restored functions for caller function.");
                }
                foreach(var function_name in saved_function.ConcreteFunctions)
                {
                    var function = concrete_functions[function_name];
                    if(function.CapturedInputs.Any(x => x is null))
                    {
                        throw new ValueError("Looks like you are trying to run a loaded " +
                            "non-Keras model that was trained using tf.distribute.experimental.ParameterServerStrategy " +
                            "with variable partitioning, which is not currently supported. Try using Keras to define your model " +
                            "if possible.");
                    }
                    if(_concrete_function_callable_with(function, inputs, false))
                    {
                        return _call_concrete_function(function, inputs);
                    }
                }
                throw new ValueError("Unexpected runtime behavior, please submit an issue to " +
                    "https://github.com/SciSharp/TensorFlow.NET/issues");
            }

            List<ConcreteFunction> concrete_function_objects = new();
            foreach(var concrete_function_name in saved_function.ConcreteFunctions)
            {
                concrete_function_objects.Add(concrete_functions[concrete_function_name]);
            }
            foreach(var cf in concrete_function_objects)
            {
                cf._set_function_spec(function_spec);
            }

            var restored_function = new RestoredFunction(restored_function_body, nameof(restored_function_body),
                function_spec, concrete_function_objects);

            return restored_function;
        }

        public static Dictionary<string, ConcreteFunction> load_function_def_library(FunctionDefLibrary library, 
            SavedObjectGraph saved_object_graph = null, string load_shared_name_suffix = null, object? wrapper_function = null)
        {
            var library_function_names = library.Function.Select(x => x.Signature.Name).Distinct();
            Dictionary<string, ConcreteFunction> functions = new();
            Dictionary<string, ConcreteFunction> renamed_functions = new();

            Graph graph;
            if (ops.executing_eagerly_outside_functions())
            {
                graph = new Graph();
            }
            else
            {
                graph = ops.get_default_graph();
            }

            if(load_shared_name_suffix is null)
            {
                load_shared_name_suffix = $"_load_{ops.uid()}";
            }

            Dictionary<ByteString, string> library_gradient_names = new();
            Dictionary<ByteString, string> new_gradient_op_types = new();
            Dictionary<string, string> gradients_to_register = new();
            foreach (var gdef in library.RegisteredGradients)
            {
                if(gdef.RegisteredOpType is not null)
                {
                    var new_op_type = custom_gradient.generate_name();
                    var old_op_type = tf.compat.as_bytes(gdef.RegisteredOpType);

                    library_gradient_names[old_op_type] = gdef.GradientFunc;
                    new_gradient_op_types[old_op_type] = new_op_type;
                    gradients_to_register[gdef.GradientFunc] = new_op_type;
                }
            }

            Dictionary<string, IEnumerable<string>> function_deps = new();
            foreach(var fdef in library.Function)
            {
                function_deps[fdef.Signature.Name] = _list_function_deps(fdef, library_function_names, library_gradient_names);
            }

            Dictionary<string, ConcreteFunction> loaded_gradients = new();
            foreach (var fdef in _sort_function_defs(library, function_deps))
            {
                var orig_name = _fix_fdef_in_place(fdef, functions, load_shared_name_suffix, new_gradient_op_types);

                object structured_input_signature = null;
                object structured_outputs = null;
                if (saved_object_graph is not null && saved_object_graph.ConcreteFunctions.ContainsKey(orig_name))
                {
                    // TODO(Rinne): deal with structured_input_signature and structured_outputs.

                    //var proto = saved_object_graph.ConcreteFunctions[orig_name];
                    //structured_input_signature = nested_structure_coder.decode_proto(proto.CanonicalizedInputSignature);
                    //structured_outputs = nested_structure_coder.decode_proto(proto.OutputSignature);
                }

                graph.as_default();
                var func_graph = function_def_lib.function_def_to_graph(fdef, structured_input_signature, structured_outputs);
                graph.Exit();

                _restore_gradient_functions(func_graph, renamed_functions, loaded_gradients);

                foreach(var dep in function_deps[orig_name])
                {
                    functions[dep].AddTograph(func_graph);
                }

                if (fdef.Attr.ContainsKey("_input_shapes"))
                {
                    fdef.Attr.Remove("_input_shapes");
                }
                var func = new ConcreteFunction(func_graph, fdef.Attr.ToDictionary(x => x.Key, x => x.Value));
                if(wrapper_function is not null)
                {
                    throw new NotImplementedException();
                }
                func.AddTograph(graph);

                functions[orig_name] = func;
                renamed_functions[func.Name] = func;
                if(func_graph.get_operations().Any(op => op.op.type == "TRTEngineOp"))
                {
                    func.AddTograph(ops.get_default_graph());
                }

                if (gradients_to_register.ContainsKey(orig_name))
                {
                    var gradient_op_type = gradients_to_register[orig_name];
                    loaded_gradients[gradient_op_type] = func;
                    ops.RegisterGradientFunction(gradient_op_type, _gen_gradient_func(func));
                }
            }
            return functions;
        }

        public static void fix_node_def(NodeDef node_def, IDictionary<string, ConcreteFunction> functions, string shared_name_suffix)
        {
            if (functions.ContainsKey(node_def.Op))
            {
                node_def.Op = functions[node_def.Op].Name;
            }
            foreach(var attr_value in node_def.Attr.Values)
            {
                if(attr_value.ValueCase == AttrValue.ValueOneofCase.Func)
                {
                    attr_value.Func.Name = functions[attr_value.Func.Name].Name;
                }
                else if(attr_value.ValueCase == AttrValue.ValueOneofCase.List)
                {
                    foreach(var fn in attr_value.List.Func)
                    {
                        fn.Name = functions[fn.Name].Name;
                    }
                }
            }

            if(node_def.Op == "HashTableV2")
            {
                if(!node_def.Attr.ContainsKey("use_node_name_sharing") || !node_def.Attr["use_node_name_sharing"].B)
                {
                    node_def.Attr["use_node_name_sharing"].B = true;
                    shared_name_suffix += $"_{ops.uid()}";
                }
            }

            var op_def = op_def_registry.GetOpDef(node_def.Op);
            if(op_def is not null)
            {
                var attr = op_def.Attr.Where(x => x.Name == "shared_name").FirstOrDefault();
                if(attr is not null)
                {
                    ByteString shared_name = null;
                    if(node_def.Attr.ContainsKey("shared_name") && node_def.Attr["shared_name"].S is not null)
                    {
                        shared_name = node_def.Attr["shared_name"].S;
                    }
                    else if(attr.DefaultValue.S is not null)
                    {
                        shared_name = tf.compat.as_bytes(attr.DefaultValue.S);
                    }
                    if(shared_name is null)
                    {
                        shared_name = tf.compat.as_bytes(node_def.Name);
                    }
                    node_def.Attr["shared_name"].S = ByteString.CopyFrom(shared_name.Concat(tf.compat.as_bytes(node_def.Name)).ToArray());
                }
            }
        }

        private static Func<Operation, Tensor[], Tensor[]> _gen_gradient_func(ConcreteFunction func)
        {
            return (unused_op, result_grads) =>
            {
                result_grads = zip(result_grads, func.func_graph.Inputs)
                    .Select((item) => item.Item1 is null ? default_gradient.zeros_like(item.Item2) : item.Item1).ToArray();
                return func.CallFlat(result_grads, func.CapturedInputs);
            };
        }

        private static void _restore_gradient_functions(FuncGraph func_graph, Dictionary<string, ConcreteFunction> renamed_functions, Dictionary<string, ConcreteFunction> loaded_gradients)
        {
            if(loaded_gradients is null || loaded_gradients.Count == 0)
            {
                foreach (var op in func_graph.get_operations())
                {
                    if (op.op.type == "StatefulPartitionedCall" || op.op.type == "PartitionedCall")
                    {
                        var function = renamed_functions[op.op.node_def.Attr["f"].Func.Name];
                        op.op._gradient_function = function._get_gradient_function();
                    }
                }
            }
            else
            {
                foreach (var op in func_graph.get_operations())
                {
                    if (op.op.type == "StatefulPartitionedCall" || op.op.type == "PartitionedCall")
                    {
                        var function = renamed_functions[op.op.node_def.Attr["f"].Func.Name];
                        op.op._gradient_function = function._get_gradient_function();
                    }
                    string gradient_op_type = null;
                    try
                    {
                        gradient_op_type = op.op.get_attr("_gradient_op_type") as string;
                    }
                    catch (InvalidArgumentError)
                    {
                        continue;
                    }
                    if (loaded_gradients.ContainsKey(gradient_op_type))
                    {
                        var grad_fn = loaded_gradients[gradient_op_type];
                        grad_fn.NumPositionArgs = op.op.inputs.Length;
                        grad_fn.ArgKeywords = op.op.inputs._inputs.Select(x => x.name);
                    }
                }
            }
        }

        private static string _fix_fdef_in_place(FunctionDef fdef, IDictionary<string, ConcreteFunction> functions, string shared_name_suffix, 
            IDictionary<ByteString, string> new_gradient_op_types)
        {
            var orig_name = fdef.Signature.Name;
            bool contains_unsaved_custom_gradients = false;

            foreach(var node_def in fdef.NodeDef)
            {
                fix_node_def(node_def, functions, shared_name_suffix);
                var op_type = _get_gradient_op_type(node_def);
                if(op_type is not null)
                {
                    if (new_gradient_op_types.ContainsKey(op_type))
                    {
                        node_def.Attr["_gradient_op_type"].S = tf.compat.as_bytes(new_gradient_op_types[op_type]);
                    }
                    else
                    {
                        contains_unsaved_custom_gradients = true;
                    }
                }
            }
            if (contains_unsaved_custom_gradients)
            {
                // TODO(Rinne): log warnings.
            }

            fdef.Signature.Name = _clean_function_name(fdef.Signature.Name);
            return orig_name;
        }

        private static string _clean_function_name(string name)
        {
            var match = Regex.Match(name, _FUNCTION_WRAPPER_NAME_REGEX);
            if(match.Success)
            {
                return match.Groups[1].Value;
            }
            else
            {
                return name;
            }
        }

        /// <summary>
        /// Return a topologic sort of FunctionDefs in a library.
        /// </summary>
        /// <param name="library"></param>
        /// <param name="function_deps"></param>
        private static IEnumerable<FunctionDef> _sort_function_defs(FunctionDefLibrary library, Dictionary<string, IEnumerable<string>> function_deps)
        {
            Dictionary<string, IList<string>> edges = new();
            Dictionary<string, int> in_count = new();
            foreach(var item in function_deps)
            {
                var fname = item.Key;
                var deps = item.Value;
                if(deps is null || deps.Count() == 0)
                {
                    in_count[fname] = 0;
                    continue;
                }
                foreach(var dep in deps)
                {
                    edges.SetDefault(dep, new List<string>()).Add(fname);
                    if (in_count.ContainsKey(fname))
                    {
                        in_count[fname]++;
                    }
                    else
                    {
                        in_count[fname] = 1;
                    }
                }
            }
            var ready = new Stack<string>(library.Function.
                Where(x => in_count[x.Signature.Name] == 0)
                .Select(x => x.Signature.Name).ToList());
            List<string> output = new();
            while(ready.Count > 0)
            {
                var node = ready.Pop();
                output.Add(node);
                if (!edges.ContainsKey(node))
                {
                    continue;
                }
                foreach(var dest in edges[node])
                {
                    in_count[dest] -= 1;
                    if (in_count[dest] == 0)
                    {
                        ready.Push(dest);
                    }
                }
            }

            if(output.Count != library.Function.Count)
            {
                var failed_to_resolve = in_count.Keys.Except(output);
                throw new ValueError($"There is a cyclic dependency between functions. " +
                    $"Could not resolve ({string.Join(", ", failed_to_resolve)}).");
            }

            var reverse = library.Function.ToDictionary(x => x.Signature.Name, x => x);
            return output.Select(x => reverse[x]);
        }

        private static IEnumerable<string> _list_function_deps(FunctionDef fdef, IEnumerable<string> library_function_names, IDictionary<ByteString, string> library_gradient_names)
        {
            HashSet<string> deps = new HashSet<string>();
            foreach(var node_def in fdef.NodeDef)
            {
                var grad_op_type = _get_gradient_op_type(node_def);
                if (library_function_names.Contains(node_def.Op))
                {
                    deps.Add(node_def.Op);
                }
                else if(grad_op_type is not null && library_gradient_names.TryGetValue(grad_op_type, out var gradient_name))
                {
                    deps.Add(gradient_name);
                }
                else
                {
                    foreach(var attr_value in node_def.Attr.Values)
                    {
                        if(attr_value.ValueCase == AttrValue.ValueOneofCase.Func)
                        {
                            deps.Add(attr_value.Func.Name);
                        }
                        else if(attr_value.ValueCase == AttrValue.ValueOneofCase.List)
                        {
                            foreach(var fn in attr_value.List.Func)
                            {
                                deps.Add(fn.Name);
                            }
                        }
                    }
                }
            }
            return deps.AsEnumerable();
        }

        private static ByteString _get_gradient_op_type(NodeDef node_def)
        {
            if(node_def.Attr.ContainsKey("_gradient_op_type") && node_def.Op != "StatefulPartitionedCall" && node_def.Op != "PartitionedCall")
            {
                return node_def.Attr["_gradient_op_type"].S;
            }
            return null;
        }

        public static ConcreteFunction setup_bare_concrete_function(SavedBareConcreteFunction saved_bare_concrete_function, 
            IDictionary<string, ConcreteFunction> concrete_functions)
        {
            var concrete_function = concrete_functions[saved_bare_concrete_function.ConcreteFunctionName];
            concrete_function.ArgKeywords = saved_bare_concrete_function.ArgumentKeywords.ToList();
            concrete_function.NumPositionArgs = saved_bare_concrete_function.AllowedPositionalArguments;

            //var function_spec = _deserialize_function_spec_as_nonmethod(saved_bare_concrete_function.FunctionSpec);
            // TODO(Rinne): set the functiona spec.
            concrete_function.AddTograph();
            return concrete_function;
        }

        private static FunctionSpec _deserialize_function_spec_as_nonmethod(FunctionSpec function_spec_proto)
        {
            // TODO(Rinne)； revise the implementation.
            return new FunctionSpec()
            {
                Fullargspec = function_spec_proto.Fullargspec,
                IsMethod = function_spec_proto.IsMethod,
                InputSignature = function_spec_proto.InputSignature,
                JitCompile = function_spec_proto.JitCompile
            };
        }

        private static Tensors _call_concrete_function(ConcreteFunction function, Tensors inputs)
        {
            // TODO(Rinne): var expected_structure = function.func_graph.structured_input_signature
            return function.CallFlat(inputs, function.CapturedInputs);
        }

        private static bool _concrete_function_callable_with(ConcreteFunction function, Tensor[] inputs, bool allow_conversion)
        {
            // TODO(Rinne): revise it.
            return function.CapturedInputs.Length + inputs.Length == function.Inputs.Length;
            //var expected_inputs = function.func_graph.Inputs;
            //foreach(var (arg, expected) in zip(inputs, expected_inputs))
            //{
            //    if(arg.Id != expected.Id)
            //    {
            //        return false;
            //    }
            //}
            //return true;
        }
    }

    public class RestoredFunction : Function
    {
        IEnumerable<ConcreteFunction> _concrete_functions;
        FunctionSpec _function_spec;
        public IEnumerable<ConcreteFunction> ConcreteFunctions => _concrete_functions;
        public RestoredFunction(Func<Tensor[], Tensor[]> function, string name, FunctionSpec function_spec,
            IEnumerable<ConcreteFunction> concrete_functions): base(function, name, auto_graph: false)
        {
            _concrete_functions = concrete_functions;
            _function_spec = function_spec;
        }

        protected override bool _run_functions_eagerly()
        {
            return false;
        }
    }
}
