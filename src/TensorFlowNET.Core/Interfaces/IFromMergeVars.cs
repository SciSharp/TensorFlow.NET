namespace Tensorflow
{
    public interface IFromMergeVars<T>
    {
        T FromMergeVars(ITensorOrTensorArray[] mergeVars);
    }
}