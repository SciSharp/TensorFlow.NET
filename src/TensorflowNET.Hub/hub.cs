using Tensorflow.Hub;

namespace Tensorflow
{
    public static class HubAPI
    {
        public static HubMethods hub { get; } = new HubMethods();
    }

    public class HubMethods
    {
        public KerasLayer KerasLayer(string handle, bool trainable = false, LoadOptions? load_options = null)
        {
            return new KerasLayer(handle, trainable, load_options);
        }
    }
}
