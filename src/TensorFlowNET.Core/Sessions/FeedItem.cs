namespace Tensorflow
{
    /// <summary>
    /// Feed dictionary item
    /// </summary>
    public class FeedItem
    {
        public object Key { get; }
        public object Value { get; }

        public FeedItem(object key, object val)
        {
            Key = key;
            Value = val;
        }

        public static implicit operator FeedItem((object, object) feed)
            => new FeedItem(feed.Item1, feed.Item2);

        public void Deconstruct(out object key, out object value)
        {
            key = Key;
            value = Value;
        }
    }
}
