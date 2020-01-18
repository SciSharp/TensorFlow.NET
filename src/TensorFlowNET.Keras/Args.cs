using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class Args
    {
        private List<object> args = new List<object>();

        public object this[int index]
        {
            get
            {
                return args.Count < index ? args[index] : null;
            }
        }

        public T Get<T>(int index)
        {
            return args.Count < index ? (T)args[index] : default(T);
        }

        public void Add<T>(T arg)
        {
            args.Add(arg);
        }
    }
}
