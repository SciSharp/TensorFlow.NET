namespace Tensorflow.Keras
{
  public class Regularizers: IRegularizerApi
  {
    public IRegularizer l1(float l1 = 0.01f)
        => new Tensorflow.Operations.Regularizers.L1(l1);
    public IRegularizer l2(float l2 = 0.01f)
        => new Tensorflow.Operations.Regularizers.L2(l2);

    //From TF source
    //# The default value for l1 and l2 are different from the value in l1_l2
    //# for backward compatibility reason. Eg, L1L2(l2=0.1) will only have l2
    //# and no l1 penalty.
    public IRegularizer l1l2(float l1 = 0.00f, float l2 = 0.00f)
        => new Tensorflow.Operations.Regularizers.L1L2(l1, l2);

    public IRegularizer L1 => l1();

    public IRegularizer L2 => l2();

    public IRegularizer L1L2 => l1l2();
  }
}
