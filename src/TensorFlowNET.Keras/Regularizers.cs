using Tensorflow.Operations.Regularizers;

namespace Tensorflow.Keras
{
  public class Regularizers: IRegularizerApi
  {
    private static Dictionary<string, IRegularizer> _nameActivationMap;

    public IRegularizer l1(float l1 = 0.01f)
        => new L1(l1);
    public IRegularizer l2(float l2 = 0.01f)
        => new L2(l2);

    //From TF source
    //# The default value for l1 and l2 are different from the value in l1_l2
    //# for backward compatibility reason. Eg, L1L2(l2=0.1) will only have l2
    //# and no l1 penalty.
    public IRegularizer l1l2(float l1 = 0.00f, float l2 = 0.00f)
        => new L1L2(l1, l2);

    static Regularizers()
    {
      _nameActivationMap = new Dictionary<string, IRegularizer>();
      _nameActivationMap["L1"] = new L1();
      _nameActivationMap["L1"] = new L2();
      _nameActivationMap["L1"] = new L1L2();
    }

    public IRegularizer L1 => l1();

    public IRegularizer L2 => l2();

    public IRegularizer L1L2 => l1l2();

    public IRegularizer GetRegularizerFromName(string name)
    {
      if (name == null)
      {
        throw new Exception($"Regularizer name cannot be null");
      }
      if (!_nameActivationMap.TryGetValue(name, out var res))
      {
        throw new Exception($"Regularizer {name} not found");
      }
      else
      {
        return res;
      }
    }
  }
}
