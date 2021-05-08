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

using NumSharp;
using NumSharp.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Tensorflow
{
    /// <summary>
    /// Binding utilities to mimic python functions.
    /// </summary>
    public static partial class Binding
    {
        public static T2 get<T1, T2>(this Dictionary<T1, T2> dict, T1 key)
            => key == null ?
                default :
            (dict.ContainsKey(key) ? dict[key] : default);

        public static void Update<T>(this IList<T> list, T element)
        {
            var index = list.IndexOf(element);
            if (index < 0)
                list.Add(element);
            else
            {
                list[index] = element;
            }
        }

        public static void difference_update<T>(this IList<T> list, IList<T> list2)
        {
            foreach(var el in list2)
            {
                if (list.Contains(el))
                    list.Remove(el);
            }
        }

        public static void add<T>(this IList<T> list, T element)
            => list.Add(element);

        public static void add<T>(this IList<T> list, IEnumerable<T> elements)
        {
            foreach (var ele in elements)
                list.Add(ele);
        }

        public static void append<T>(this IList<T> list, T element)
            => list.Insert(list.Count, element);

        public static void append<T>(this IList<T> list, IList<T> elements)
        {
            for (int i = 0; i < elements.Count(); i++)
                list.Insert(list.Count, elements[i]);
        }

        public static T[] concat<T>(this IList<T> list1, IList<T> list2)
        {
            var list = new List<T>();
            list.AddRange(list1);
            list.AddRange(list2);
            return list.ToArray();
        }

        public static void extend<T>(this List<T> list, IEnumerable<T> elements)
            => list.AddRange(elements);

        private static string _tostring(object obj)
        {
            switch (obj)
            {
                case NDArray nd:
                    return nd.ToString(false);
                case Array arr:
                    if (arr.Rank != 1 || arr.GetType().GetElementType()?.IsArray == true)
                        arr = Arrays.Flatten(arr);
                    var objs = toObjectArray(arr);
                    return $"[{string.Join(", ", objs.Select(_tostring))}]";
                default:
                    return obj?.ToString() ?? "null";
            }

            object[] toObjectArray(Array arr)
            {
                var len = arr.LongLength;
                var ret = new object[len];
                for (long i = 0; i < len; i++)
                {
                    ret[i] = arr.GetValue(i);
                }

                return ret;
            }
        }

        private static TextWriter writer = null;

        public static TextWriter tf_output_redirect { 
            set
            {
                var originWriter = writer ?? Console.Out;
                originWriter.Flush();
                if (originWriter is StringWriter)
                    (originWriter as StringWriter).GetStringBuilder().Clear();
                writer = value;
            }
            get
            {
                return writer ?? Console.Out;
            }
        }

        public static void print(object obj)
        {
            tf_output_redirect.WriteLine(_tostring(obj));
        }

        public static void print(string format, params object[] objects)
        {
            if (!format.Contains("{}"))
            {
                tf_output_redirect.WriteLine(format + " " + string.Join(" ", objects.Select(x => x.ToString())));
                return;
            }

            foreach (var obj in objects)
            {

            }

            tf_output_redirect.WriteLine(format);
        }

        public static int len(object a)
        {
            switch (a)
            {
                case Tensor tensor:
                    return tensor.shape[0];
                case Tensors arr:
                    return arr.Length;
                case Array arr:
                    return arr.Length;
                case IList arr:
                    return arr.Count;
                case ICollection arr:
                    return arr.Count;
                case NDArray ndArray:
                    return ndArray.ndim == 0 ? 1 : ndArray.shape[0];
                case IEnumerable enumerable:
                    return enumerable.OfType<object>().Count();
                case TensorShape arr:
                    return arr.ndim;
            }
            throw new NotImplementedException("len() not implemented for type: " + a.GetType());
        }

        public static float min(float a, float b)
            => Math.Min(a, b);

        public static int max(int a, int b)
            => Math.Max(a, b);

        public static T[] list<T>(IEnumerable<T> list)
            => list.ToArray();

        public static IEnumerable<int> range(int end)
        {
            return Enumerable.Range(0, end);
        }

        public static IEnumerable<int> range(int start, int end)
        {
            return Enumerable.Range(start, end - start);
        }

        public static IEnumerable<T> reversed<T>(IList<T> values)
        {
            var len = values.Count;
            for (int i = len - 1; i >= 0; i--)
                yield return values[i];
        }

        public static T New<T>() where T : ITensorFlowObject, new()
        {
            var instance = new T();
            instance.__init__();
            return instance;
        }

        [DebuggerStepThrough]
        public static void tf_with(ITensorFlowObject py, Action<ITensorFlowObject> action)
        {
            try
            {
                py.__enter__();
                action(py);
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerStepThrough]
        public static void tf_with<T>(T py, Action<T> action) where T : ITensorFlowObject
        {
            try
            {
                py.__enter__();
                action(py);
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerStepThrough]
        public static TOut tf_with<TIn, TOut>(TIn py, Func<TIn, TOut> action) where TIn : ITensorFlowObject
        {
            try
            {
                py.__enter__();
                return action(py);
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        public static float time()
        {
            return (float)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds;
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>((T1, T1) t1, (T2, T2) t2)
        {
            for (int i = 0; i < 2; i++)
            {
                if (i == 0)
                    yield return (t1.Item1, t2.Item1);
                else
                    yield return (t1.Item2, t2.Item2);
            }
        }

        public static IEnumerable<(T, T)> zip<T>(NDArray t1, NDArray t2)
            where T : unmanaged
        {
            var a = t1.AsIterator<T>();
            var b = t2.AsIterator<T>();
            while (a.HasNext() && b.HasNext())
                yield return (a.MoveNext(), b.MoveNext());
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IList<T1> t1, IList<T2> t2)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i]);
        }

        public static IEnumerable<(T1, T2, T3)> zip<T1, T2, T3>(IList<T1> t1, IList<T2> t2, IList<T3> t3)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i], t3[i]);
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(NDArray t1, NDArray t2)
            where T1 : unmanaged
            where T2 : unmanaged
        {
            var a = t1.AsIterator<T1>();
            var b = t2.AsIterator<T2>();
            while (a.HasNext() && b.HasNext())
                yield return (a.MoveNext(), b.MoveNext());
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IEnumerable<T1> e1, IEnumerable<T2> e2)
        {
            return e1.Zip(e2, (t1, t2) => (t1, t2));
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(Dictionary<TKey, TValue> values)
        {
            foreach (var item in values)
                yield return (item.Key, item.Value);
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(KeyValuePair<TKey, TValue>[] values)
        {
            var len = values.Length;
            for (var i = 0; i < len; i++)
            {
                var item = values[i];
                yield return (item.Key, item.Value);
            }
        }

        public static IEnumerable<(int, T)> enumerate<T>(IList<T> values)
        {
            var len = values.Count;
            for (int i = 0; i < len; i++)
                yield return (i, values[i]);
        }
        
        public static IEnumerable<(int, T)> enumerate<T>(IEnumerable<T> values, int start = 0, int step = 1)
        {
            int i = 0;
            foreach (var val in values)
            {
                if (i++ < start)
                    continue;

                yield return (i - 1, val);
            }
        }

        [DebuggerStepThrough]
        public static Dictionary<string, object> ConvertToDict(object dyn)
        {
            var dictionary = new Dictionary<string, object>();
            foreach (PropertyDescriptor propertyDescriptor in TypeDescriptor.GetProperties(dyn))
            {
                object obj = propertyDescriptor.GetValue(dyn);
                string name = propertyDescriptor.Name;
                dictionary.Add(name, obj);
            }
            return dictionary;
        }


        public static bool all(IEnumerable enumerable)
        {
            foreach (var e1 in enumerable)
            {
                if (!Convert.ToBoolean(e1))
                    return false;
            }
            return true;
        }

        public static bool any(IEnumerable enumerable)
        {
            foreach (var e1 in enumerable)
            {
                if (Convert.ToBoolean(e1))
                    return true;
            }
            return false;
        }

        public static double sum(IEnumerable enumerable)
        {
            var typedef = new Type[] { typeof(double), typeof(int), typeof(float) };
            var sum = 0.0d;
            foreach (var e1 in enumerable)
            {
                if (!typedef.Contains(e1.GetType()))
                    throw new Exception("Numeric array expected");
                sum += (double)e1;
            }
            return sum;
        }

        public static float sum(IEnumerable<float> enumerable)
            => enumerable.Sum();

        public static int sum(IEnumerable<int> enumerable)
            => enumerable.Sum();

        public static double sum<TKey, TValue>(Dictionary<TKey, TValue> values)
        {
            return sum(values.Keys);
        }

        public static IEnumerable<double> slice(double start, double end, double step = 1)
        {
            for (double i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<float> slice(float start, float end, float step = 1)
        {
            for (float i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<int> slice(int start, int end, int step = 1)
        {
            for (int i = start; i < end; i += step)
                yield return i;
        }

        public static IEnumerable<int> slice(int range)
        {
            for (int i = 0; i < range; i++)
                yield return i;
        }

        public static bool hasattr(object obj, string key)
        {
            var __type__ = (obj).GetType();

            var __member__ = __type__.GetMembers();
            var __memberobject__ = __type__.GetMember(key);
            return (__memberobject__.Length > 0) ? true : false;
        }

        public static IEnumerable TupleToEnumerable(object tuple)
        {
            Type t = tuple.GetType();
            if (t.IsGenericType && (t.FullName.StartsWith("System.Tuple") || t.FullName.StartsWith("System.ValueTuple")))
            {
                var flds = t.GetFields();
                for (int i = 0; i < flds.Length; i++)
                {
                    yield return flds[i].GetValue(tuple);
                }
            }
            else
            {
                throw new System.Exception("Expected Tuple.");
            }
        }

        public static bool isinstance(object Item1, Type Item2)
        {
            return Item1.GetType() == Item2;
        }

        public static bool isinstance(object Item1, object tuple)
        {
            foreach (var t in TupleToEnumerable(tuple))
                if (isinstance(Item1, (Type)t))
                    return true;
            return false;
        }

        public static bool issubset<T>(this IEnumerable<T> subset, IEnumerable<T> src)
        {
            bool issubset = true;
            foreach (var element in subset)
            {
                if (!src.Contains(element))
                {
                    issubset = false;
                    continue;
                }
            }

            return true;
        }

        public static void extendleft<T>(this Queue<T> queue, IEnumerable<T> elements)
        {
            foreach (var element in elements.Reverse())
                queue.Enqueue(element);
        }

        public static bool empty<T>(this Queue<T> queue)
            => queue.Count == 0;

        public static TValue SetDefault<TKey, TValue>(this Dictionary<TKey, TValue> dic, TKey key, TValue defaultValue)
        {
            if (dic.ContainsKey(key))
                return dic[key];

            dic[key] = defaultValue;
            return defaultValue;
        }

        public static TValue Get<TKey, TValue>(this Dictionary<TKey, TValue> dic, TKey key, TValue defaultValue)
        {
            if (dic.ContainsKey(key))
                return dic[key];

            return defaultValue;
        }
    }
}
