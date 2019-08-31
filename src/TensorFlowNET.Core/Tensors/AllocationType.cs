namespace Tensorflow
{
    /// <summary>
    ///     Used internally to 
    /// </summary>
    public enum AllocationType
    {
        None = 0,
        /// <summary>
        ///     Allocation was done by passing in a pointer, might be also holding reference to a C# object.
        /// </summary>
        FromPointer = 1,
        /// <summary>
        ///     Allocation was done by calling c_api.TF_AllocateTensor or TF decided it has to copy data during c_api.TF_NewTensor. <br></br>
        ///     Deallocation is handled solely by Tensorflow.
        /// </summary>
        Tensorflow = 2,
        /// <summary>
        ///     Allocation was done by Marshal.AllocateHGlobal
        /// </summary>
        Marshal = 3,
        /// <summary>
        ///     Allocation was done by GCHandle.Alloc
        /// </summary>
        GCHandle = 4,
    }
}