using System.Collections.Generic;

namespace Tensorflow.Util
{
    public class UnorderedMap<Tk, Tv> : Dictionary<Tk, Tv>
    {
        /// <summary>
        /// Avoid null when accessing not existed element
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public new Tv this[Tk key]
        {
            get
            {
                if (!ContainsKey(key))
                    Add(key, default);

                return base[key];
            }

            set
            {
                base[key] = value;
            }
        }

        public Tv SetDefault(Tk key, Tv default_value)
        {
            if(TryGetValue(key, out var res))
            {
                return res;
            }
            else
            {
                base[key] = default_value;
                return base[key];
            }
        }

        public void push_back(Tk key, Tv value)
            => this[key] = value;

        public void emplace(Tk key, Tv value)
            => this[key] = value;

        public bool find(Tk key)
            => ContainsKey(key);

        public void erase(Tk key)
            => Remove(key);

        public bool find(Tk key, out Tv value)
        {
            if (ContainsKey(key))
            {
                value = this[key];
                return true;
            }
            else
            {
                value = default(Tv);
                return false;
            }
        }
    }

    public class UnorderedMapEnumerable<Tk, Tv> : UnorderedMap<Tk, Tv>
        where Tv : new()
    {
        public new Tv this[Tk key]
        {
            get
            {
                if (!ContainsKey(key))
                    Add(key, new Tv());

                return base[key];
            }

            set
            {
                base[key] = value;
            }
        }
    }
}
