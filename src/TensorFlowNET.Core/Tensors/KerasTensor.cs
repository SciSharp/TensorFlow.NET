namespace Tensorflow.Keras.Engine;

/// <summary>
/// A representation of a Keras in/output during Functional API construction.
/// </summary>
public class KerasTensor
{
    private Tensor _tensor;
    public void SetTensor(Tensors tensor) 
        => _tensor = tensor;

    private TensorSpec _type_spec;
    private string _name;

    public KerasTensor(TensorSpec type_spec, string name = null)
    {
        _type_spec = type_spec;
        _name = name;
    }

    public static KerasTensor from_tensor(Tensor tensor)
    {
        var type_spec = tensor.ToTensorSpec();
        var kt = new KerasTensor(type_spec, name: tensor.name);
        kt.SetTensor(tensor);
        return kt;
    }

    public static implicit operator Tensors(KerasTensor kt)
        => kt._tensor;

    public static implicit operator Tensor(KerasTensor kt)
        => kt._tensor;

    public static implicit operator KerasTensor(Tensor tensor)
        => from_tensor(tensor);

    public static implicit operator KerasTensor(Tensors tensors)
        => from_tensor(tensors.First());
}
