using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Losses;

public abstract class LossFunctionWrapper : Loss
{
    public LossFunctionWrapper(string reduction = ReductionV2.AUTO,
        string name = null,
        bool from_logits = false)
        : base(reduction: reduction,
              name: name,
              from_logits: from_logits)
    { }
}
