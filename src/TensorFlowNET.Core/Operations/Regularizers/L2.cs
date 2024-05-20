using System;

using Tensorflow.Keras;

namespace Tensorflow.Operations.Regularizers
{
  public class L2 : IRegularizer
  {
    float _l2;
    private readonly Dictionary<string, object> _config;

    public string ClassName => "L2";
    public virtual IDictionary<string, object> Config => _config;

    public L2(float l2 = 0.01f)
    {
      //  l2 = 0.01 if l2 is None else l2
      //  validate_float_arg(l2, name = "l2")
      //  self.l2 = l2
      this._l2 = l2;

      _config = new();
      _config["l2"] = _l2;
    }


    public Tensor Apply(RegularizerArgs args)
    {
      //return self.l2 * ops.sum(ops.square(x))
      return _l2 * math_ops.reduce_sum(math_ops.square(args.X));
    }
  }
}
