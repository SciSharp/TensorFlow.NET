namespace Tensorflow
{
    public interface IPackable<T>
    {
        T Pack(object[] sequences);
    }
}
