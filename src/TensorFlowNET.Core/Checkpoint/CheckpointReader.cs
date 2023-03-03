namespace Tensorflow.Checkpoint;

public class CheckpointReader
{
    private SafeCheckpointReaderHandle _handle;
    public Dictionary<string, TF_DataType> VariableToDataTypeMap { get; set; }
    public Dictionary<string, Shape> VariableToShapeMap { get; set; }

    public CheckpointReader(string filename)
    {
        Status status = new Status();
        VariableToDataTypeMap = new Dictionary<string, TF_DataType>();
        VariableToShapeMap = new Dictionary<string, Shape>();
        _handle = c_api.TF_NewCheckpointReader(filename, status);
        status.Check(true);
        ReadAllShapeAndType();
    }

    public int HasTensor(string name)
        => c_api.TF_CheckpointReaderHasTensor(_handle, name);

    /// <summary>
    /// Get the variable name.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public string GetVariable(int index)
        => c_api.StringPiece(c_api.TF_CheckpointReaderGetVariable(_handle, index));

    public int Size()
        => c_api.TF_CheckpointReaderSize(_handle);

    public TF_DataType GetVariableDataType(string name)
        => c_api.TF_CheckpointReaderGetVariableDataType(_handle, name);

    public Shape GetVariableShape(string name)
    {
        int num_dims = GetVariableNumDims(name);
        long[] dims = new long[num_dims];
        Status status = new Status();
        c_api.TF_CheckpointReaderGetVariableShape(_handle, name, dims, num_dims, status);
        status.Check(true);
        return new Shape(dims);
    }

    public int GetVariableNumDims(string name)
        => c_api.TF_CheckpointReaderGetVariableNumDims(_handle, name);

    public unsafe Tensor GetTensor(string name, TF_DataType dtype = TF_DataType.DtInvalid)
    {
        Status status = new Status();
        var tensor = c_api.TF_CheckpointReaderGetTensor(_handle, name, status);
        status.Check(true);
        return new Tensor(tensor);
    }

    private void ReadAllShapeAndType()
    {
        int size = Size();
        for(int i = 0; i < size; i++)
        {
            var name = GetVariable(i);
            var shape = GetVariableShape(name);
            var dtype = GetVariableDataType(name);
            VariableToDataTypeMap[name] = dtype;
            VariableToShapeMap[name] = shape;
        }
    }
}
