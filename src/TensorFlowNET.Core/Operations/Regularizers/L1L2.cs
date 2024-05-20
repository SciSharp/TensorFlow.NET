using System;

using Tensorflow.Keras;

namespace Tensorflow.Operations.Regularizers
{
  public class L1L2 : IRegularizer
  {
    float _l1;
    float _l2;
    private readonly Dictionary<string, object> _config;

    public string ClassName => "L1L2";
    public virtual IDictionary<string, object> Config => _config;

    public L1L2(float l1 = 0.0f, float l2 = 0.0f)
    {
      //l1 = 0.0 if l1 is None else l1
      //l2 = 0.0 if l2 is None else l2
      //  validate_float_arg(l1, name = "l1")
      //  validate_float_arg(l2, name = "l2")

      //  self.l1 = l1
      //  self.l2 = l2
      this._l1 = l1;
      this._l2 = l2;

      _config = new();
      _config["l1"] = l1;
      _config["l2"] = l2;
    }

    public Tensor Apply(RegularizerArgs args)
    {
        //regularization = ops.convert_to_tensor(0.0, dtype = x.dtype)
        //if self.l1:
        //    regularization += self.l1 * ops.sum(ops.absolute(x))
        //if self.l2:
        //    regularization += self.l2 * ops.sum(ops.square(x))
        //return regularization

        Tensor regularization = tf.constant(0.0, args.X.dtype);
        regularization += _l1 * math_ops.reduce_sum(math_ops.abs(args.X));
        regularization += _l2 * math_ops.reduce_sum(math_ops.square(args.X));
        return regularization;
    }
  }
}
