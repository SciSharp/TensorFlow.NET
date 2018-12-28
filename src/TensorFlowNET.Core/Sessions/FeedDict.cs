using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class FeedDict : IEnumerable
    {
        private Dictionary<Tensor, object> feed_dict;

        public FeedDict()
        {
            feed_dict = new Dictionary<Tensor, object>();
        }

        public object this[Tensor feed]
        {
            get
            {
                return feed_dict[feed];
            }

            set
            {
                feed_dict[feed] = value;
            }
        }

        public FeedDict Add(Tensor feed, object value)
        {
            feed_dict.Add(feed, value);
            return this;
        }

        public IEnumerator GetEnumerator()
        {
            foreach (KeyValuePair<Tensor, object> feed in feed_dict)
            {
                yield return new FeedValue
                {
                    feed = feed.Key,
                    feed_val = feed.Value
                };
            }
        }

        public Dictionary<Tensor, object> items()
        {
            return feed_dict;
        }
    }

    public struct FeedValue
    {
        public Tensor feed { get; set; }
        public object feed_val { get; set; }
    }
}
