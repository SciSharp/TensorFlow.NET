using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Train;

namespace Tensorflow;

public static class SignatureSerializationUtils
{
    internal static readonly string DEFAULT_SIGNATURE_ATTR = "_default_save_signature";
    internal static readonly string SIGNATURE_ATTRIBUTE_NAME = "signatures";
    internal static readonly int _NUM_DISPLAY_NORMALIZED_SIGNATURES = 5;
    public static SignatureMap create_signature_map(IDictionary<string, Trackable> signatures)
    {
        var signature_map = new SignatureMap();
        foreach (var pair in signatures)
        {
            var name = pair.Key;
            var func = pair.Value;
            Debug.Assert(func is ConcreteFunction);
            // TODO: assert the `func.structured_outputs` and arg_keywords.
            signature_map._add_signature(name, (ConcreteFunction)func);
        }

        return signature_map;
    }

    public static ConcreteFunction find_function_to_export(AugmentedGraphView graph_view)
    {
        var children = graph_view.list_children(graph_view.Root);
        List<Trackable> possible_signatures = new();
        foreach (var item in children)
        {
            var name = item.Name;
            var child = item.Refer;
            if(child is not (Function or ConcreteFunction))
            {
                continue;
            }
            if(name == DEFAULT_SIGNATURE_ATTR)
            {
                Debug.Assert(child is ConcreteFunction);
                return (ConcreteFunction)child;
            }
            ConcreteFunction concrete = get_signature(child);
            if(concrete is not null && valid_signature(concrete))
            {
                possible_signatures.Add(concrete);
            }
        }

        if(possible_signatures.Count == 1)
        {
            var signature = get_signature(possible_signatures[0]);
            if(signature is not null && valid_signature(signature))
            {
                return signature;
            }
        }
        return null;
    }

    private static ConcreteFunction get_signature(Trackable function)
    {
        // TODO: implement it.
        return null;
    }

    private static bool valid_signature(ConcreteFunction concreate_function)
    {
        // TODO: implement it.
        return false;
    }
}

public class SignatureMap: Trackable
{
    private Dictionary<string, Trackable> _signatures;

    public SignatureMap()
    {
        _signatures = new();
    }

    public void _add_signature(string name, ConcreteFunction concrete_function)
    {
        _signatures[name] = concrete_function;
    }
    
    public void _add_signature(string name, Function concrete_function)
    {
        _signatures[name] = concrete_function;
    }

    public override IDictionary<string, Trackable> _trackable_children(SaveType save_type, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
    {
        if (save_type != SaveType.SAVEDMODEL)
        {
            return new Dictionary<string, Trackable>();
        }

        return _signatures.Where(x => x.Value is Function or ConcreteFunction).ToDictionary(x => x.Key, x => x.Value);
    }
}
