namespace Tensorflow
{
    /// <summary>
    /// Mode for variable access within a variable scope.
    /// </summary>
    public enum _ReuseMode
    {
        NOT_REUSE = 0,
        // Indicates that variables are to be fetched if they already exist or
        // otherwise created.
        AUTO_REUSE = 1
    }
}
