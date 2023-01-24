using System;

namespace Tensorflow.Keras.Saving.SavedModel;

public partial class KerasSavedModelUtils
{
    public static bool ShouldHaveTraces { get; internal set; } = true;

    public static SaveOptionsContext keras_option_scope(bool save_traces)
    {
        var res = new SaveOptionsContext(ShouldHaveTraces);
        ShouldHaveTraces = save_traces;
        return res;
    }
}

/// <summary>
/// Implementation of this class is different with that of python.
/// But it could be used with `using` the same as `with` of python.
/// </summary>
public class SaveOptionsContext: IDisposable
{
    public bool _old_value;
    public SaveOptionsContext(bool old_value)
    {
        _old_value = old_value;
    }

    public void Dispose()
    {
        KerasSavedModelUtils.ShouldHaveTraces = _old_value;
    }
}