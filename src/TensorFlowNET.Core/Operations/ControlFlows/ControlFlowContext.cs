/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Operations.ControlFlows;
using static Tensorflow.Binding;
using static Tensorflow.ControlFlowContextDef;
using util = Tensorflow.control_flow_util;

namespace Tensorflow.Operations
{
    /// <summary>
    /// The base class for control flow context.
    /// 
    /// The usage pattern is a sequence of(Enter, Exit) followed by a final
    /// ExitResult.
    /// 
    /// We maintain the following state for control flow contexts during graph
    /// construction:
    /// 1. graph has _control_flow_context: the current context used to
    /// construct new nodes.Changed by ctxt.Enter() and ctxt.Exit()
    /// 2. op has _control_flow_context: the context to which the op belongs.
    /// Set at the time the op is created.Immutable.
    /// 3. A ControlFlowContext has _outer_context: the context in which this
    /// context is created.Set at the time a context is created.Immutable.
    /// 4. A ControlFlowContext has _context_stack.
    /// Pushed and popped by ctxt.Enter() and ctxt.Exit()
    /// </summary>
    public abstract class ControlFlowContext : ITensorFlowObject
    {
        /// <summary>
        /// The predicate tensor in this branch
        /// </summary>
        protected Tensor _pivot;
        public Tensor pivot => _pivot;

        /// <summary>
        /// The boolean tensor for the cond predicate
        /// </summary>
        protected Tensor _pred;
        public Tensor pred => _pred;

        /// <summary>
        /// 0 or 1 representing this branch
        /// </summary>
        protected int _branch;
        public int branch => _branch;

        protected Stack<ControlFlowContext> _context_stack;
        protected ControlFlowContext _outer_context;

        /// <summary>
        /// The keys are the names of tensors referenced by but external to this
        /// context. Each value is the Tensor that should be used by this context to
        /// access the key value (e.g. a switch output guarding a cond input value).
        /// </summary>
        protected Dictionary<string, ITensorOrOperation> _external_values;

        public ControlFlowContext()
        {
            _context_stack = new Stack<ControlFlowContext>();
            _external_values = new Dictionary<string, ITensorOrOperation>();
        }

        public string Name { get => _name; }
        protected string _name;

        public void __init__(ValuesDef values_def = null, string import_scope = null)
        {
            _outer_context = ops.get_default_graph()._get_control_flow_context();
            if (values_def != null)
                _init_values_from_proto(values_def, import_scope: import_scope);
            else
            {
                _values = new HashSet<string>();
                _external_values = new Dictionary<string, ITensorOrOperation>();
            }

        }

        public void __enter__()
        {
        }

        /// <summary>
        /// Initializes values and external_values from `ValuesDef` protocol buffer.
        /// </summary>
        /// <param name="values_def"></param>
        /// <param name="import_scope"></param>
        protected void _init_values_from_proto(ValuesDef values_def, string import_scope = null)
        {
            _external_values = new Dictionary<string, ITensorOrOperation>();
            foreach (var value in values_def.Values)
                _values.Add(value);
            var g = ops.get_default_graph();
            foreach (var value in values_def.ExternalValues)
            {
                var k = ops.prepend_name_scope(value.Key, import_scope);
                var v = value.Value;
                _external_values[k] = g.as_graph_element(ops.prepend_name_scope(v, import_scope));
            }

            var op_names = _values.Where(x => !_external_values.ContainsKey(x))
                .Select(x => x.Split(':')[0])
                .ToArray();

            foreach (var op in op_names)
                (g.as_graph_element(op) as Operation)._set_control_flow_context(this);
        }

        public void __exit__()
        {
        }

        /// <summary>
        /// Enter this control flow context.
        /// </summary>
        public virtual void Enter()
        {
            var graph = ops.get_default_graph();
            _context_stack.Push(graph._get_control_flow_context());
            graph._set_control_flow_context(this);
        }

        /// <summary>
        /// Exit this control flow context.
        /// </summary>
        public virtual void Exit()
        {
            var graph = ops.get_default_graph();
            var last_context = _context_stack.Pop();
            graph._set_control_flow_context(last_context);
        }

        public void ExitResult(Tensor[] result)
        {
            if (_outer_context != null)
            {
                throw new NotImplementedException("ExitResult");
            }
        }

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        public virtual void AddOp(Operation op)
        {
            _AddOpInternal(op);
        }

        public ControlFlowContext outer_context { get { return _outer_context; } }
        public HashSet<string> values => _values;

        public virtual GradLoopState grad_state => throw new NotImplementedException("abstract method");

        public virtual bool back_prop => throw new NotImplementedException("abstract method");

        /// <summary>
        /// Add `val` to the current context and its outer context recursively.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public virtual Tensor AddValue(Tensor val)
        {
            // to be overridden
            return null;
        }

        public void AddName(string name)
        {
            _values.Add(name);
        }

        /// <summary>
        /// Notifies a scope about an operator added to an inner scope.
        /// </summary>
        /// <param name="op"></param>
        public virtual void AddInnerOp(Operation op)
        {
            if (_outer_context != null)
                _outer_context.AddInnerOp(op);
        }

        protected HashSet<string> _values = new HashSet<string>();

        /// <summary>
        /// Add `op` to the current context.
        /// </summary>
        protected virtual void _AddOpInternal(Operation op)
        {
            if (op == null)
            {
                throw new NotImplementedException("");
            }
            else
            {
                foreach (var index in range(len(op.inputs)))
                {
                    var x = op.inputs[index];
                    var real_x = AddValue(x);
                    if (real_x != x)
                        op._update_input(index, real_x);
                }
            }
        }

        protected bool OpInContext(Operation op)
        {
            return IsContainingContext(op._get_control_flow_context(), this);
        }

        /// <summary>
        /// Returns true if `maybe_containing_ctxt` is or contains `ctxt`.
        /// </summary>
        public static bool IsContainingContext(ControlFlowContext ctxt, ControlFlowContext maybe_containing_ctxt)
        {
            while (ctxt != maybe_containing_ctxt)
            {
                if (ctxt == null)
                    return false;
                ctxt = ctxt.outer_context;
            }
            return true;
        }

        protected virtual bool _IsInOuterContext(Operation op)
        {
            throw new NotImplementedException("_IsInOuterContext");
        }

        /// <summary>
        /// Remove any external control dependency on this op.
        /// </summary>
        /// <param name="op"></param>
        protected virtual (Operation[], Operation[]) _RemoveExternalControlEdges(Operation op)
        {
            var while_ctxt = GetWhileContext();

            var internal_control_inputs = new List<Operation>();
            // A control input of `op` is internal if it is in the same while
            // loop context as the enclosing while loop context of self.
            if (while_ctxt == null)
            {
                internal_control_inputs = op.control_inputs.ToList();
            }
            else
            {
                foreach (Operation x in op.control_inputs)
                {
                    var ctxt = util.GetOutputContext(x);
                    if (ctxt != null && ctxt.GetWhileContext() == while_ctxt)
                        internal_control_inputs.append(x);
                }
            }

            var external_control_inputs = new List<Operation>();
            if (len(internal_control_inputs) != len(op.control_inputs))
                throw new NotImplementedException("");

            return (internal_control_inputs.ToArray(), external_control_inputs.ToArray());
        }

        /// <summary>
        /// Return the while context containing this context
        /// </summary>
        public virtual WhileContext GetWhileContext()
        {
            if (_outer_context != null)
                return _outer_context.GetWhileContext();
            return null;
        }

        /// <summary>
        /// Deserializes `context_def` into the appropriate ControlFlowContext.
        /// </summary>
        /// <param name="context_def">ControlFlowContextDef proto</param>
        /// <param name="import_scope">Name scope to add</param>
        /// <returns>A ControlFlowContext subclass</returns>
        protected ControlFlowContext from_control_flow_context_def(ControlFlowContextDef context_def, string import_scope = "")
        {
            switch (context_def.CtxtCase)
            {
                case CtxtOneofCase.CondCtxt:
                    return new CondContext().from_proto(context_def.CondCtxt, import_scope: import_scope);
                case CtxtOneofCase.WhileCtxt:
                    return new WhileContext().from_proto(context_def.WhileCtxt, import_scope: import_scope);
            }

            throw new NotImplementedException($"Unknown ControlFlowContextDef field: {context_def.CtxtCase}");
        }

        public virtual bool IsWhileContext()
            => false;

        public virtual bool IsCondContext()
            => false;

        public object to_proto()
        {
            throw new NotImplementedException();
        }


        public void Dispose()
        {
        }

        public void __init__()
        {

        }

        public void __del__()
        {

        }
    }
}
