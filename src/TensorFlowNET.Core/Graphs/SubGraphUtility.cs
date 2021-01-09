using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    public class SubGraphUtility
    {
        /// <summary>
        /// Copies the tensor and all its inputs recursively to the outer graph.
        /// </summary>
        /// <param name="tensors"></param>
        /// <param name="graph"></param>
        /// <param name="add_sources"></param>
        /// <param name="handle_captures"></param>
        /// <param name="base_graph"></param>
        /// <returns></returns>
        public static Dictionary<ITensorOrOperation, Operation> lift_to_graph(Tensors init_tensors, 
            FuncGraph graph, 
            List<Tensor> sources,
            bool add_sources = false,
            bool handle_captures = false,
            Graph base_graph = null,
            Dictionary<ITensorOrOperation, Operation> op_map = null)
        {
            base_graph = base_graph ?? init_tensors[0].graph;
            op_map = op_map ?? new Dictionary<ITensorOrOperation, Operation>();
            var visited_ops = sources.Select(x => x.op).ToList();
            foreach (var init_tensor in init_tensors)
            {
                var src = map_subgraph(init_tensor, sources, visited_ops, add_sources);
                sources.AddRange(src);
            }

            var ops_to_copy = new List<Operation>();
            var marked_ops = new List<Operation>();
            var ops_to_visit = new Stack<Operation>(init_tensors.Select(x => x.op));
            var unvisited_ops = new List<Operation>(ops_to_visit.ToList());
            while (unvisited_ops.Count > 0)
            {
                while(ops_to_visit.Count > 0)
                {
                    var op = ops_to_visit.Pop();
                    if (marked_ops.Contains(op))
                        continue;
                    marked_ops.Add(op);
                    ops_to_copy.append(op);
                    foreach(var inp in op.inputs)
                    {

                    }
                }
                // difference_update
                unvisited_ops.difference_update(marked_ops);
                if (unvisited_ops.Count > 0)
                    ops_to_visit.Push(unvisited_ops.Last());
            }

            // When lifting from one FuncGraph to another, we will need to capture the
            // relevant tensors as well.
            var inverse_captures = new Dictionary<Tensor, Tensor>();
            Tensor[] internal_captures = null;
            if (base_graph is FuncGraph base_func_graph)
            {
                var captures = base_func_graph.captures;
                foreach (var (external_capture, internal_capture) in captures)
                    inverse_captures[internal_capture] = external_capture;
                internal_captures = base_func_graph.internal_captures;
            }

            graph.as_default();
            var source_ops = new List<Operation>();
            // Add the sources in the same order as the original graph.
            foreach (var s in internal_captures)
            {
                if (sources.Contains(s))
                {
                    sources.Remove(s);
                    source_ops.Add(s.op);
                    _copy_source(s: s,
                        graph: graph,
                        op_map: op_map,
                        handle_captures: handle_captures,
                        inverse_captures: inverse_captures,
                        base_graph: base_graph);
                }
            }

            foreach(var op in reversed(ops_to_copy))
            {
                if (source_ops.Contains(op) || op_map.ContainsKey(op))
                    continue;
                _copy_non_source(op, graph, op_map, base_graph);
            }

            graph.Exit();

            return op_map;
        }

        static void _copy_source(Tensor s, 
            FuncGraph graph,
            Dictionary<ITensorOrOperation, Operation> op_map, 
            bool handle_captures,
            Dictionary<Tensor, Tensor> inverse_captures,
            Graph base_graph)
        {
            Tensor copied_placeholder = null;
            if (handle_captures && inverse_captures.ContainsKey(s))
                copied_placeholder = graph.capture(inverse_captures[s], name: s.op.name);
            else
                throw new NotImplementedException("");
            op_map[s] = copied_placeholder;
            // Add an entry for the op of the source tensor so that if there are any nodes
            // depending on that op via control dependencies it can work correctly.
            op_map[s.op] = copied_placeholder.op;
        }

        static void _copy_non_source(Operation op, FuncGraph graph, Dictionary<ITensorOrOperation, Operation> op_map, Graph base_graph)
        {
            Operation copied_op = null;
            var copied_inputs = new Tensors();
            tf_with(ops.control_dependencies(new object[] { op }), delegate
            {
                // Create a new op in the destination graph if it doesn't exist before.
                var attrs = new Dictionary<string, AttrValue>();
                foreach (var attr_def in op.node_def.Attr)
                    attrs[attr_def.Key] = attr_def.Value;

                copied_op = graph.create_op(op.type,
                    copied_inputs,
                    dtypes: op.outputs.Select(x => x.dtype).ToArray(),
                    attrs: attrs,
                    name: op.name);
            });
            op_map[op] = copied_op;
            foreach (var (i, o) in enumerate(op.outputs))
                op_map[o] = copied_op.outputs[i];
        }

        /// <summary>
        /// Walk a Graph and capture the subgraph between init_tensor and sources.
        /// </summary>
        /// <param name="init_tensor"></param>
        /// <param name="add_sources"></param>
        public static List<Tensor> map_subgraph(Tensor init_tensor,
            List<Tensor> sources,
            List<Operation> visited_ops, 
            bool add_sources)
        {
            var ops_to_visit = new Stack<Operation>();
            ops_to_visit.Push(init_tensor.op);
            var extra_sources = new List<Tensor>();
            while (ops_to_visit.Count > 0)
            {
                var op = ops_to_visit.Pop();
                if (visited_ops.Contains(op))
                    continue;
                visited_ops.Add(op);
                bool should_raise = false;
                if (should_raise)
                    throw new RuntimeError($"Unable to lift tensor {init_tensor.name}.");
                if(op.type == "Placeholder")
                {
                    extra_sources.AddRange(op.outputs);
                }
                foreach(var inp in op.inputs)
                {

                }
            }
            return extra_sources;
        }
    }
}
