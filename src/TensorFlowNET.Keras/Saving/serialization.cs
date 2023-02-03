using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Tensorflow.Keras.Saving.SavedModel;

namespace Tensorflow.Keras.Saving
{
    // TODO: make it thread safe.
    public class SharedObjectSavingScope: IDisposable
    {
        private class WeakReferenceEqualityComparer: IEqualityComparer<WeakReference<object>>
        {
            public bool Equals(WeakReference<object> x, WeakReference<object> y)
            {
                if(!x.TryGetTarget(out var tx))
                {
                    return false;
                }
                if(!y.TryGetTarget(out var ty))
                {
                    return false;
                }
                return tx.Equals(ty);
            }
            public int GetHashCode(WeakReference<object> obj)
            {
                if (!obj.TryGetTarget(out var w))
                {
                    return 0;
                }
                return w.GetHashCode();
            }
        }
        private static SharedObjectSavingScope? _instance = null;
        private readonly Dictionary<WeakReference<object>, int> _shared_object_ids= new Dictionary<WeakReference<object>, int>();
        private int _currentId = 0;
        /// <summary>
        /// record how many times the scope is nested.
        /// </summary>
        private int _nestedDepth = 0;
        private SharedObjectSavingScope()
        {

        }

        public static SharedObjectSavingScope Enter()
        {
            if(_instance is not null)
            {
                _instance._nestedDepth++;
                return _instance;
            }
            else
            {
                _instance = new SharedObjectSavingScope();
                _instance._nestedDepth++;
                return _instance;
            }
        }

        public static SharedObjectSavingScope GetScope()
        {
            return _instance;
        }

        public int GetId(object? obj)
        {
            if(obj is null)
            {
                return _currentId++;
            }
            var maybe_key = _shared_object_ids.Keys.SingleOrDefault(x => new WeakReferenceEqualityComparer().Equals(x, new WeakReference<object>(obj)));
            if (maybe_key is not null)
            {
                return _shared_object_ids[maybe_key];
            }
            _shared_object_ids[new WeakReference<object>(obj)] = _currentId++;
            return _currentId;
        }

        public void Dispose()
        {
            _nestedDepth--;
            if(_nestedDepth== 0)
            {
                _instance = null;
            }
        }
    }

    public static class serialize_utils
    {
        public static readonly string SHARED_OBJECT_KEY = "shared_object_id";
        /// <summary>
        /// Returns the serialization of the class with the given config.
        /// </summary>
        /// <param name="class_name"></param>
        /// <param name="config"></param>
        /// <param name="obj"></param>
        /// <param name="shared_object_id"></param>
        /// <returns></returns>
        public static JObject serialize_keras_class_and_config(string class_name, JToken config, object? obj = null, int? shared_object_id = null)
        {
            JObject res = new JObject();
            res["class_name"] = class_name;
            res["config"] = config;

            if(shared_object_id is not null)
            {
                res[SHARED_OBJECT_KEY] = shared_object_id!;
            }

            var scope = SharedObjectSavingScope.GetScope();
            if(scope is not null && obj is not null)
            {
                res[SHARED_OBJECT_KEY] = scope.GetId(obj);
            }

            return res;
        }
    }
}
