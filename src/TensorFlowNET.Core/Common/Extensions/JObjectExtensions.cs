using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Extensions
{
    public static class JObjectExtensions
    {
        public static T? TryGetOrReturnNull<T>(this JObject obj, string key)
        {
            var res = obj[key];
            if (res is null)
            {
                return default;
            }
            else
            {
                return res.ToObject<T>();
            }
        }
    }
}
