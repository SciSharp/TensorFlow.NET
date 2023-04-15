using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using Tensorflow.Graphs;

namespace Tensorflow.Functions
{
    public class TracingCompiler
    {
        Func<Tensor[], Tensor[]> _csharp_function;
        //FunctionSpec _function_spec;
        internal string _name;
        bool _autograph;
        Dictionary<string, ConcreteFunction> _function_cache;
        Dictionary<string, AttrValue> _function_attributes;
        int _tracing_count;
        
        public TracingCompiler(Func<Tensor[], Tensor[]> csharp_function, string name, object? input_signatures = null, 
            Dictionary<string, AttrValue> attributes = null, bool autograph = true, object? autograph_options = null, 
            bool reduce_retracing = false, bool capture_by_value = false)
        {
            _csharp_function = csharp_function;
            bool pure_function = attributes is not null && attributes.Count > 0 && attributes.ContainsKey(monomorphic_function_utils.IMPLEMENTS_ATTRIBUTE_NAME);
            _name = name;
            _autograph = autograph;
            _function_attributes = attributes ?? new Dictionary<string, AttrValue>();
            _function_cache = new Dictionary<string, ConcreteFunction>();
            _tracing_count = 0;
        }

        public Tensor[] Apply(Tensor[] inputs)
        {
            // TODO(Rinne): add lock here.
            var (concrete_function, filtered_flat_args) = _maybe_define_function(inputs);
            return concrete_function.CallFlat(filtered_flat_args, concrete_function.CapturedInputs);
        }

        internal ConcreteFunction _get_concrete_function_internal_garbage_collected(Tensor[] args)
        {
            var (concrete_function, _) = _maybe_define_concrete_function(args);
            return concrete_function;
        }

        private (ConcreteFunction, Tensor[]) _maybe_define_concrete_function(Tensor[] args)
        {
            return _maybe_define_function(args);
        }

        private (ConcreteFunction, Tensor[]) _maybe_define_function(Tensor[] args)
        {
            var lookup_func_key = make_cache_key(args);
            if(_function_cache.TryGetValue(lookup_func_key, out var concrete_function))
            {
                return (concrete_function, args);
            }
            concrete_function = _create_concrete_function(args);
            _function_cache[lookup_func_key] = concrete_function;
            return (concrete_function, args);
        }

        private ConcreteFunction _create_concrete_function(Tensor[] args)
        {
            _tracing_count++;
            
            int arglen = args.Length;
            var concrete_function = new ConcreteFunction(FuncGraph.func_graph_from_func(
                    _name, x => _csharp_function(x.Where(y => y is Tensor).Select(y => (Tensor)y).ToArray()), 
                    args, new Dictionary<string, object>(), autograph: _autograph
                ), _function_attributes);
            return concrete_function;
        }

        private static string make_cache_key(Tensor[] inputs)
        {
            //string res = "";
            //foreach (var input in inputs)
            //{
            //    res += $"{input.name}_{input.Id}";
            //}
            return inputs.Length.ToString();
        }
    }
}
