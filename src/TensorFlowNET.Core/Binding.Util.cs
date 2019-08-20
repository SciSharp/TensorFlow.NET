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
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;

namespace Tensorflow
{
    /// <summary>
    /// Binding utilities to mimic python functions.
    /// </summary>
    public static partial class Binding
    {
        public static void print(object obj)
        {
            Console.WriteLine(obj.ToString());
        }

        public static int len(object a)
        {
            switch (a)
            {
                case Array arr:
                    return arr.Length;
                case IList arr:
                    return arr.Count;
                case ICollection arr:
                    return arr.Count;
                case NDArray ndArray:
                    return ndArray.shape[0];
                case IEnumerable enumerable:
                    return enumerable.OfType<object>().Count();
            }
            throw new NotImplementedException("len() not implemented for type: " + a.GetType());
        }

        public static IEnumerable<int> range(int end)
        {
            return Enumerable.Range(0, end);
        }

        public static IEnumerable<int> range(int start, int end)
        {
            return Enumerable.Range(start, end - start);
        }

        public static T New<T>() where T : IObjectLife, new()
        {
            var instance = new T();
            instance.__init__();
            return instance;
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static void tf_with(IObjectLife py, Action<IObjectLife> action)
        {
            try
            {
                py.__enter__();
                action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static void tf_with<T>(T py, Action<T> action) where T : IObjectLife
        {
            try
            {
                py.__enter__();
                action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static TOut tf_with<TIn, TOut>(TIn py, Func<TIn, TOut> action) where TIn : IObjectLife
        {
            try
            {
                py.__enter__();
                return action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return default(TOut);
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

        public static IEnumerable<(T, T)> zip<T>(NDArray t1, NDArray t2)
            where T : unmanaged
        {
            var a = t1.AsIterator<T>();
            var b = t2.AsIterator<T>();
            while (a.HasNext())
                yield return (a.MoveNext(), b.MoveNext());
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IList<T1> t1, IList<T2> t2)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i]);
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(NDArray t1, NDArray t2) 
            where T1: unmanaged
            where T2: unmanaged
        {
            var a = t1.AsIterator<T1>();
            var b = t2.AsIterator<T2>();
            while(a.HasNext())
                yield return (a.MoveNext(), b.MoveNext());
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IEnumerable<T1> e1, IEnumerable<T2> e2)
        {
            var iter2 = e2.GetEnumerator();
            foreach (var v1 in e1)
            {
                iter2.MoveNext();
                var v2 = iter2.Current;
                yield return (v1, v2);
            }
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(Dictionary<TKey, TValue> values)
        {
            foreach (var item in values)
                yield return (item.Key, item.Value);
        }

        public static IEnumerable<(TKey, TValue)> enumerate<TKey, TValue>(KeyValuePair<TKey, TValue>[] values)
        {
            foreach (var item in values)
                yield return (item.Key, item.Value);
        }

        public static IEnumerable<(int, T)> enumerate<T>(IList<T> values)
        {
            for (int i = 0; i < values.Count; i++)
                yield return (i, values[i]);
        }

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

        public delegate object __object__(params object[] args);

        public static __object__ getattr(object obj, string key, params Type[] ___parameter_type__)
        {
            var __dyn_obj__ = obj.GetType().GetMember(key);
            if (__dyn_obj__.Length == 0)
                throw new Exception("The object \"" + nameof(obj) + "\" doesnot have a defination \"" + key + "\"");
            var __type__ = __dyn_obj__[0];
            if (__type__.MemberType == System.Reflection.MemberTypes.Method)
            {
                try
                {
                    var __method__ = (___parameter_type__.Length > 0) ? obj.GetType().GetMethod(key, ___parameter_type__) : obj.GetType().GetMethod(key);
                    return (object[] args) => __method__.Invoke(obj, args);
                }
                catch (System.Reflection.AmbiguousMatchException ex)
                {
                    throw new Exception("AmbigousFunctionMatchFound : (Probable cause : Function Overloading) Please add parameter types of the function.");
                }
            }
            else if (__type__.MemberType == System.Reflection.MemberTypes.Field)
            {
                var __field__ = obj.GetType().GetField(key).GetValue(obj);
                return (object[] args) => { return __field__; };
            }
            else if (__type__.MemberType == System.Reflection.MemberTypes.Property)
            {
                var __property__ = obj.GetType().GetProperty(key).GetValue(obj);
                return (object[] args) => { return __property__; };
            }
            return (object[] args) => { return "NaN"; };
        }

        public static IEnumerable TupleToEnumerable(object tuple)
        {
            Type t = tuple.GetType();
            if(t.IsGenericType && (t.FullName.StartsWith("System.Tuple") || t.FullName.StartsWith("System.ValueTuple")))
            {
                var flds = t.GetFields();
                for(int i = 0; i < flds.Length;i++)
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
            var tup = TupleToEnumerable(tuple);
            foreach(var t in tup)
            {
                if(isinstance(Item1, (Type)t))
                    return true;
            }
            return false;
        }
    }
}
