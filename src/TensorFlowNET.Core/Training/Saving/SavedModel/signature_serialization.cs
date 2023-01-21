using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Train;

namespace Tensorflow;

public class SignatureMap: Trackable
{
    private Dictionary<string, Function> _signatures;
    private Dictionary<string, ConcreteFunction> _concrete_signatures;

    public SignatureMap()
    {
        _signatures = new();
    }

    public void _add_signature(string name, ConcreteFunction concrete_function)
    {
        _concrete_signatures[name] = concrete_function;
    }
    
    public void _add_signature(string name, Function concrete_function)
    {
        _signatures[name] = concrete_function;
    }

    public override IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, object>? cache = null)
    {
        if (save_type != SaveType.SAVEDMODEL)
        {
            return new Dictionary<string, Trackable>();
        }

        Dictionary<string, Trackable> res = _signatures.ToDictionary(x => x.Key, x => (Trackable)x.Value);
        foreach (var pair in _concrete_signatures)
        {
            res[pair.Key] = pair.Value;
        }

        return res;
    }

    public static SignatureMap create_signature_map(IDictionary<string, ConcreteFunction> signatures)
    {
        var signature_map = new SignatureMap();
        foreach (var pair in signatures)
        {
            var name = pair.Key;
            var func = pair.Value;
            // TODO: assert the arg_keywords
            signature_map._add_signature(name, func);
        }

        return signature_map;
    }
}