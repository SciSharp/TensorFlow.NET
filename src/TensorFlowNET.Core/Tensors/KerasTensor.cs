namespace Tensorflow.Keras.Engine;

/// <summary>
/// A representation of a Keras in/output during Functional API construction.
/// </summary>
public class KerasTensor
{
    private Tensors _inferred_value;
    public Tensors inferred_value 
    {
        get => _inferred_value;
        set => _inferred_value = value;
    }

    private string _name;
    private TensorSpec _type_spec;
    public Shape shape => _type_spec.shape;
    public TF_DataType dtype => _type_spec.dtype;

    public KerasTensor(TensorSpec type_spec, string name = null)
    {
        _type_spec = type_spec;
        _name = name;
    }

    public static KerasTensor from_tensor(Tensor tensor)
    {
        var type_spec = tensor.ToTensorSpec();
        var kt = new KerasTensor(type_spec, name: tensor.name);
        kt.inferred_value = tensor;
        return kt;
    }

    public override string ToString()
        => _inferred_value.Length switch
        {
            > 1 => "[" + string.Join(", ", _inferred_value.Select(x => $"<KerasTensor: shape={x.shape} dtype={x.dtype}>")) + "]",
            1 => $"<KerasTensor: shape={_inferred_value.shape} dtype={_inferred_value.dtype}>",
            _ => _inferred_value.ToString(),
        };

    public static implicit operator Tensors(KerasTensor kt)
        => kt._inferred_value;

    public static implicit operator Tensor(KerasTensor kt)
        => kt._inferred_value;

    public static implicit operator KerasTensor(Tensor tensor)
        => from_tensor(tensor);

    public static implicit operator KerasTensor(Tensors tensors)
        => from_tensor(tensors.First());
}
