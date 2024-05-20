using System;

using Tensorflow.Keras;

namespace Tensorflow.Operations.Regularizers
{
  public class L1 : IRegularizer
  {
    float _l1;
    private readonly Dictionary<string, object> _config;

    public string ClassName => "L1";
    public virtual IDictionary<string, object> Config => _config;

    public L1(float l1 = 0.01f)
    {
      //  l1 = 0.01 if l1 is None else l1
      //  validate_float_arg(l1, name = "l1")
      //  self.l1 = ops.convert_to_tensor(l1)
      this._l1 = l1;

      _config = new();
      _config["l1"] = _l1;
    }


    public Tensor Apply(RegularizerArgs args)
    {
      //return self.l1 * ops.sum(ops.absolute(x))
      return _l1 * math_ops.reduce_sum(math_ops.abs(args.X));
    }
  }
}
