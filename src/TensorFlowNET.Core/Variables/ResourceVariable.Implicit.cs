namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public static implicit operator _VariableScopeStore(ResourceVariable variable)
        {
            return null;
        }

        public static implicit operator ResourceVariable(_VariableScopeStore store)
        {
            return null;
        }

        public static implicit operator Tensor(ResourceVariable var)
        {
            return null;
        }

        public static implicit operator ResourceVariable(Tensor var)
        {
            return null;
        }

        public static implicit operator RefVariable(ResourceVariable var)
        {
            return null;
        }
    }
}
