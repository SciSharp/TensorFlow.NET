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

        public static (T, T) zip<T>(IList<T> t1, IList<T> t2, int index = 0)
        {
            return (t1[index], t2[index]);
        }

        public static (T, T) zip<T>(NDArray t1, NDArray t2, int index = 0)
        {
            return (t1.Data<T>(index), t2.Data<T>(index));
        }
    }

    public interface IPython : IDisposable
    {
        void __enter__();

        void __exit__();
    }
}
