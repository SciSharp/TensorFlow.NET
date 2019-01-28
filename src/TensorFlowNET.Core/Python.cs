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
    }

    public interface IPython : IDisposable
    {
        void __enter__();

        void __exit__();
    }
}
