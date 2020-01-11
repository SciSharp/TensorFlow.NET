using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class KwArgs
    {
        private Dictionary<string, object> args = new Dictionary<string, object>();

        public object this[string name]
        {
            get
            {
                return args.ContainsKey(name) ? args[name] : null;
            }
            set
            {
                args[name] = value;
            }
        }

        public T Get<T>(string name)
        {
            if (!args.ContainsKey(name))
                return default(T);

            return (T)args[name];
        }

        public static explicit operator KwArgs(ValueTuple<string, object>[] p)
        {
            KwArgs kwArgs = new KwArgs();
            kwArgs.args = new Dictionary<string, object>();
            foreach (var item in p)
            {
                kwArgs.args[item.Item1] = item.Item2;
            }

            return kwArgs;
        }
    }
}
