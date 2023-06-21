using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Eager;
using Tensorflow.Framework;
using Tensorflow.Framework.Models;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace Tensorflow.Operations
{
    class _OperationWithOutputs : Operation
    {
        public _OperationWithOutputs(IntPtr handle, Graph g = null)
        {
            _handle = handle;
            _graph = g;
            _outputs = null;
            g._add_op(this);
        }
    }
    internal class while_v2
    {
        public static Tensor[] while_loop(Func<Tensors, Tensor> cond,
            Func<Tensors, Tensors> body,
            Tensors loop_vars,
            int maximum_iterations = -1, 
            int parallel_iterations = 10,
            string name = null, 
            bool back_prop = true, 
            bool return_same_structure = true)
        {
            var orig_loop_vars = loop_vars;
            var flat_orig_loop_vars = orig_loop_vars.Flatten().ToArray();
            int len_orig_loop_vars = orig_loop_vars.Length;

            loop_vars = _tensor_array_to_flow(loop_vars);
            loop_vars = Nest.MapStructure(x => _convert_to_tensor_or_indexed_slices(x), loop_vars).ToTensors();

            var loop_vars_signature = Nest.MapStructure(x => new TensorSpec(x.shape, x.dtype), loop_vars);

            var flat_shape_invariants = Nest.Flatten(loop_vars_signature).Select(x => x.shape).ToArray();

            if(string.IsNullOrEmpty(name))
            {
                name = "while";
            }

            return tf_with<ITensorFlowObject, Tensor[]>(ops.name_scope(name), nameScopeWhile =>
            {
                string scope = (nameScopeWhile as ops.NameScope).scope_name;
                string cond_name = control_flow_util.unique_fn_name(scope, "cond");
                string body_name = control_flow_util.unique_fn_name(scope, "body");

                var maximum_iterations_loop_var = _build_maximum_iterations_loop_var(maximum_iterations);
                var loop_counter = constant_op.constant(0, maximum_iterations == -1 ? TF_DataType.DtInvalid : maximum_iterations_loop_var.dtype,
                    name: "loop_counter");
                loop_vars = new Tensor[] { loop_counter, maximum_iterations_loop_var }.Concat(loop_vars).ToArray();

                var func_graph_signature = new TensorSpec[] {TensorSpec.FromTensor(loop_counter),TensorSpec.FromTensor(maximum_iterations_loop_var)}
                    .Concat(loop_vars_signature.Flatten()).ToArray();

                // TODO(Rinne): possible wrong implemenation here.
                var add_control_dependencies = false;

                object[] wrapped_cond(object[] inputs)
                {
                    Tensor loop_counter = (Tensor)inputs[0];
                    Tensor maximum_iterations_arg = (Tensor)inputs[1];
                    Tensor[] args = inputs.Skip(2).Select(x => (Tensor)x).ToArray();
                    var pred = cond(_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args));
                    if(pred.shape.IsNull || pred.shape.ndim > 0)
                    {
                        pred = array_ops.squeeze(pred);
                    }
                    if(maximum_iterations == -1)
                    {
                        return new object[] { pred };
                    }
                    else
                    {
                        return new object[] { math_ops.logical_and(loop_counter < maximum_iterations_arg, pred) };
                    }
                }

                var cond_graph = FuncGraph.func_graph_from_func(cond_name, wrapped_cond, null,
                    null, signature: func_graph_signature, add_control_dependencies: add_control_dependencies);

                bool stateful_parallelism = false;

                object[] wrapped_body(object[] inputs)
                {
                    Tensor loop_counter = (Tensor)inputs[0];
                    Tensor maximum_iterations_arg = (Tensor)inputs[1];
                    Tensor[] args = inputs.Skip(2).Select(x => (Tensor)x).ToArray();

                    _copy_handle_data(loop_vars.Flatten().Skip(2), args);

                    foreach(var t in cond_graph.external_captures)
                    {
                        var graph = (FuncGraph)(ops.get_default_graph());
                        graph.capture(t);
                    }

                    var outputs = body(_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args));
                    outputs = _tensor_array_to_flow(outputs);

                    return new object[] { loop_counter + 1, maximum_iterations_arg }.Concat(outputs).ToArray();
                }

                var body_graph = FuncGraph.func_graph_from_func(body_name, wrapped_body, null, null, func_graph_signature,
                    add_control_dependencies: add_control_dependencies, acd_record_initial_resource_uses: stateful_parallelism);

                // TODO(Rinne): possible wrong implementation here.
                NestList<Tensors> loop_vars_list = new(new Tensors[] { loop_vars, body_graph.external_captures.ToTensors() });
                body_graph.Outputs.AddRange(body_graph.internal_captures);
                
                cond_graph.as_default();
                int num_cond_captures = cond_graph.external_captures.Length;
                Debug.Assert(cond_graph.external_captures.SequenceEqual(body_graph.external_captures.Take(num_cond_captures).ToArray()));
                _duplicate_body_captures_in_cond(cond_graph, body_graph.external_captures.Skip(num_cond_captures).ToArray());
                cond_graph.Exit();

                int first_loop_var_index = 2;

                int num_flattened_oututs = orig_loop_vars.Length;
                int num_original_outputs = body_graph.Outputs.Length;
                if (back_prop && control_flow_util.output_all_intermediates())
                {
                    var intermediate_tensors = _get_intermediates(body_graph);

                    foreach(var intermediate_tensor in intermediate_tensors)
                    {
                        var tensor_list = list_ops.empty_tensor_list(intermediate_tensor.shape, intermediate_tensor.dtype, maximum_iterations);
                        loop_vars_list.Values.Add(tensor_list);

                        cond_graph.as_default();
                        cond_graph.capture(tensor_list);
                        cond_graph.Exit();

                        body_graph.as_default();
                        var appended_tensor_list = gen_ops.tensor_list_push_back(tensor_list, intermediate_tensor);
                        body_graph.Outputs.Add(appended_tensor_list);
                        body_graph.Exit();
                    }
                }

                List<Tensor> flattened_loop_vars = new();
                foreach(var item in loop_vars_list.Values)
                {
                    flattened_loop_vars.AddRange(item.Flatten());
                }
                // skip the check

                // TODO(Rinne): deal with control dependencies
                var output_shapes = body_graph.Outputs.Select(t => t.shape).ToArray();
                var span = new Span<Shape>(output_shapes).Slice(first_loop_var_index, num_flattened_oututs);
                for(int i = 0; i < span.Length; i++)
                {
                    span[i] = flat_shape_invariants[i];
                }

                Tensor[] outputs = _build_while_op(flattened_loop_vars.ToArray(), cond_graph, body_graph, output_shapes, parallel_iterations,
                    (nameScopeWhile as ops.NameScope).scope_name, num_original_outputs, stateful_parallelism);

                if (!ops.get_default_graph().building_function)
                {
                    outputs = outputs.Select(t => array_ops.identity(t)).ToArray();
                }

                var output_loop_vars = outputs.Skip(first_loop_var_index).Take(num_flattened_oututs).ToArray();

                if (!back_prop)
                {
                    output_loop_vars = output_loop_vars.Select(t => array_ops.stop_gradient(t)).ToArray();
                }
                outputs = _pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, output_loop_vars);

                return outputs;
            });
        }

        private static Tensors _tensor_array_to_flow(Tensors loop_vars)
        {
            if(loop_vars.NestType == NestType.Node)
            {
                if(loop_vars.NodeValue is FakeTensorByTensorArray fake)
                {
                    return new Tensors(fake.TensorArray.flow);
                }
                else
                {
                    return new Tensors(loop_vars.NodeValue!);
                }
            }
            else if(loop_vars.NestType == NestType.List)
            {
                List<INestStructure<Tensor>> list = new();
                foreach(var item in loop_vars.ListValue!)
                {
                    if(item.NestType == NestType.Node)
                    {
                        var nested = item.AsNest();
                        if (nested.NodeValue is FakeTensorByTensorArray fake)
                        {
                            list.Add(new Nest<Tensor>(fake.TensorArray.flow));
                        }
                        else
                        {
                            list.Add(new Nest<Tensor>(nested.NodeValue!));
                        }
                    }
                    else
                    {
                        list.Add(new Nest<Tensor>(item.AsNest()));
                    }
                }
                return Tensors.FromNest(new Nest<Tensor>(list));
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        private static Tensor[] _build_while_op(Tensor[] loop_vars, FuncGraph cond_graph, FuncGraph body_graph,
            Shape[] output_shapes, int parallel_iterations, string name, int num_original_outputs, bool stateful_parallelism)
        {
            var cond_stateful_ops = cond_graph.get_operations().Select(x => x.op);
            var body_stateful_ops = body_graph.get_operations().Select(x => x.op);

            bool is_stateful = cond_stateful_ops.Count() > 0 || body_stateful_ops.Count() > 0;

            Tensor[] _make_op(Tensor[] inputs)
            {
                Tensor[] outputs;
                if (is_stateful)
                {
                    outputs = gen_functional_ops._while(
                            inputs,
                            control_flow_util.create_new_tf_function(cond_graph),
                            control_flow_util.create_new_tf_function(body_graph),
                            output_shapes,
                            parallel_iterations,
                            name
                        );
                }
                else
                {
                    outputs = gen_functional_ops.stateless_while(
                            inputs,
                            control_flow_util.create_new_tf_function(cond_graph),
                            control_flow_util.create_new_tf_function(body_graph),
                            output_shapes,
                            parallel_iterations,
                            name
                        );
                }
                var (while_op, tensors) = control_flow_util.get_op_and_outputs(outputs);
                _copy_handle_data(body_graph.Outputs, tensors);
                _set_read_only_resource_inputs_attr(while_op, new FuncGraph[]{cond_graph, body_graph});
                while_op._set_attr("_num_original_outputs", new AttrValue() { I = num_original_outputs });
                while_op._set_attr("_stateful_parallelism", new AttrValue() { B = stateful_parallelism });

                cond_graph.outer_graph = ops.get_default_graph();
                body_graph.outer_graph = ops.get_default_graph();
                // TODO(Rinne): set the two graphs to while_op
                return tensors;
            }

            return control_flow_util.run_as_function_for_tape_gradients(_make_op, loop_vars);
        }

        /// <summary>
        /// Sets the list of resource inputs which are read-only. This is used by AutomaticControlDependencies.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="branch_graphs"></param>
        private static void _set_read_only_resource_inputs_attr(Operation op, FuncGraph[] branch_graphs)
        {
            List<int> read_only_indices = Enumerable.Range(0, op.inputs.Length).ToList();
            foreach(var branch_graph in branch_graphs)
            {
                if (read_only_indices.Count == 0)
                {
                    break;
                }
                var branch_read_only_indices = auto_control_deps_utils.get_read_only_resource_input_indices_graph(branch_graph);
                read_only_indices = read_only_indices.Intersect(branch_read_only_indices).ToList();
            }
            AttrValue.Types.ListValue listValue = new();
            listValue.I.AddRange(read_only_indices.OrderBy(x => x).Select(x => (long)x));
            op._set_attr(auto_control_deps_utils.READ_ONLY_RESOURCE_INPUTS_ATTR, new AttrValue()
            {
                List = listValue
            });
        }

        private static Tensors _pack_sequence_as<T>(INestStructure<T> loop_vars_signature, Tensor[] flat_orig_loop_vars, Tensor[] loop_vars)
        {
            var flattened_loop_vars = zip(loop_vars, flat_orig_loop_vars).Select<(Tensor, Tensor), Tensor>(item =>
            {
                var (flow, y) = item;
                if (y is FakeTensorByTensorArray ta)
                {
                    return new FakeTensorByTensorArray(tensor_array_ops.build_ta_with_new_flow(ta.TensorArray, flow));
                }
                else
                {
                    return flow;
                }
            }).ToArray();
            return Nest.PackSequenceAs(loop_vars_signature, flattened_loop_vars).ToTensors();
        }

        private static Tensor[] _get_intermediates(FuncGraph func_graph)
        {
            List<Tensor> intermediates = new();
            var reversed_captures = func_graph.captures.ToDictionary(x => x.Item2, x => x.Item1);

            foreach(var op in func_graph.get_operations())
            {
                Debug.Assert(op is Operation);
                var oper = (Operation)op;
                if(oper.type == "Identity" || oper.type == "MutexLock")
                {
                    continue;
                }
                foreach(var o in  op.outputs)
                {
                    if(o != func_graph.Inputs[0] && o.dtype != dtypes.resource && !reversed_captures.ContainsKey(o))
                    {
                        intermediates.Add(o);
                    }
                }
            }
            return intermediates.ToArray();
        }

        private static void _duplicate_body_captures_in_cond(FuncGraph cond_graph, Tensor[] body_graph_captures)
        {
            var types = body_graph_captures.Select(t => t.dtype).ToList();
            var c_graph = cond_graph.c_graph;
            var placeholders = types.Select(x => CreatePlaceholder(c_graph, _build_cond_placeholders_name_prefix(cond_graph), x)).ToList();

            var placeholder_ops = placeholders.Select(ph => new _OperationWithOutputs(ph.oper, cond_graph)).ToList();

            List<Tensor> tensors = new();
            foreach(var (op, ph, dtype) in zip(placeholder_ops, placeholders, types))
            {
                var tensor = Tensor._create_with_tf_output(op, 0, dtype, ph);
                op._outputs = new Tensor[] { tensor };
                tensors.Add(tensor);
            }

            var tuples = zip(body_graph_captures, tensors).ToList();
            var keys = body_graph_captures.Select(t => t.Id).ToList();
            cond_graph._captures.Update(zip(keys, tuples).ToDictionary(x => x.Item1, x => x.Item2));
            cond_graph.Inputs.AddRange(tensors);
        }

        private static TF_Output CreatePlaceholder(SafeGraphHandle graph, string name, TF_DataType dtype)
        {
            var desc = c_api.TF_NewOperation(graph, "Placeholder", name);
            c_api.TF_SetAttrType(desc, "dtype", dtype);
            var op = c_api.TF_FinishOperation(desc, tf.Status);
            tf.Status.Check(true);
            var output = new TF_Output();
            output.oper = op;
            output.index = 0;
            return output;
        }

        private static string _build_cond_placeholders_name_prefix(FuncGraph cond_graph)
        {
            return cond_graph.unique_name(cond_graph.Name + "___redundant_placeholder");
        }

        private static Tensor _convert_to_tensor_or_indexed_slices(Tensor value)
        {
            return ops.convert_to_tensor(value, as_ref: false);
        }

        private static Tensor _build_maximum_iterations_loop_var(int maximum_iterations = -1)
        {
            return ops.convert_to_tensor(maximum_iterations, dtypes.int32, "maximum_iterations");
        }

        private static void _copy_handle_data(IEnumerable<Tensor> src_tensors, IEnumerable<Tensor> dst_tensors)
        {
            foreach(var (src_t, dst_t) in zip(src_tensors, dst_tensors))
            {
                handle_data_util.copy_handle_data(src_t, dst_t);
            }
        }
    }
}
