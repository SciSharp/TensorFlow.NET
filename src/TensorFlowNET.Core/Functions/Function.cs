using System;
using Tensorflow.Functions;
using Tensorflow.Train;

namespace Tensorflow
{
    public class Function: Trackable
    {
#pragma warning disable CS0169 // The field 'Function._handle' is never used
        private IntPtr _handle;
#pragma warning restore CS0169 // The field 'Function._handle' is never used

        protected Func<Tensors, Tensors> _function;
        protected ConcreteFunction _concrete_variable_creation_fn;
        protected bool _auto_graph;
        public string Name { get; set; }
        public Function(Func<Tensors, Tensors> function, 
            string name, bool auto_graph = true)
        {
            _function = function;
            Name = name;
            _auto_graph = auto_graph;
        }

        public virtual Tensors Apply(Tensors inputs)
        {
            if (_run_functions_eagerly())
            {
                return _function(inputs);
            }

            var result = _call(inputs);
            return result;
        }

        protected virtual Tensors _call(Tensors inputs)
        {
            _initialize();

            return _concrete_variable_creation_fn.CallFlat(inputs,
                _concrete_variable_creation_fn.CapturedInputs);
        }

        protected virtual bool _run_functions_eagerly()
        {
            return false;
        }

        private void _initialize()
        {

        }
    }
}
