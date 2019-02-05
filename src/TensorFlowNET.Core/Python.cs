using NumSharp.Core;
using System;
using System.Collections.Generic;
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
                throw ex;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        public static void with<T>(IPython py, Action<T> action) where T : IPython
        {
            try
            {
                py.__enter__();
                action((T)py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw ex;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        public static TOut with<TIn, TOut>(IPython py, Func<TIn, TOut> action) where TIn : IPython
        {
            try
            {
                py.__enter__();
                return action((TIn)py);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw ex;
            }
            finally
            {
                py.__exit__();
                py.Dispose();
            }
        }

        public static IEnumerable<(T, T)> zip<T>(NDArray t1, NDArray t2)
        {
            int index = 0;
            yield return(t1.Data<T>(index), t2.Data<T>(index));
        }

        public static IEnumerable<(T, T)> zip<T>(IList<T> t1, IList<T> t2)
        {
            for (int i = 0; i < t1.Count; i++)
                yield return (t1[i], t2[i]);
        }

        public static IEnumerable<(int, T)> enumerate<T>(IList<T> values)
        {
            for (int i = 0; i < values.Count; i++)
                yield return (i, values[i]);
        }
    }

    public interface IPython : IDisposable
    {
        void __enter__();

        void __exit__();
    }
}
