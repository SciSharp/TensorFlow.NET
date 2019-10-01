namespace Tensorflow.Operations.Losses
{
    public class Util
    {
        public static void add_loss(Tensor loss, string loss_collection = "losses")
        {
            if (!string.IsNullOrEmpty(loss_collection))
                ops.add_to_collection(loss_collection, loss);
        }
    }
}
