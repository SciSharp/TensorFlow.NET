namespace Tensorflow.Framework.Models
{
    public class ScopedTFGraph : Graph
    {
        public ScopedTFGraph() : base()
        {

        }

        ~ScopedTFGraph()
        {
            base.Dispose();
        }
    }
}
