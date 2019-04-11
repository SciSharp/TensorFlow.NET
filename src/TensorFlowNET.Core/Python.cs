using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Mapping C# functions to Python
    /// </summary>
    public class Python
    {
        protected void print(object obj)
        {
            Console.WriteLine(obj.ToString());
        }

        protected int len<T>(IEnumerable<T> a)
            => a.Count();

        protected IEnumerable<int> range(int end)
        {
            return Enumerable.Range(0, end);
        }

        public static T New<T>(object args) where T : IPyClass
        {
            var instance = Activator.CreateInstance<T>();

            instance.__init__(instance, args);

            return instance;
        }

        [DebuggerNonUserCode()] // with "Just My Code" enabled this lets the debugger break at the origin of the exception
        public static void with(IPython py, Action<IPython> action)
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
        public static void with<T>(T py, Action<T> action) where T : IPython
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
        public static TOut with<TIn, TOut>(TIn py, Func<TIn, TOut> action) where TIn : IPython
        {
            try
            {
                py.__enter__();
                return action(py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
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
        {
            for (int i = 0; i < t1.size; i++)
                yield return (t1.Data<T>(i), t2.Data<T>(i));
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(IList<T1> t1, IList<T2> t2)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i]);
        }

        public static IEnumerable<(T1, T2)> zip<T1, T2>(NDArray t1, NDArray t2)
        {
            for (int i = 0; i < t1.size; i++)
                yield return (t1.Data<T1>(i), t2.Data<T2>(i));
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
    }

    public interface IPython : IDisposable
    {
        void __enter__();

        void __exit__();
    }

    public class PyObject<T> where T : IPyClass
    {
        public T Instance { get; set; }
    }
}
