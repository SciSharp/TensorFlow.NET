namespace Tensorflow.Operations
{
    public class MergeOutput
    {
        Tensor output;
        Tensor value_index;
        public MergeOutput(Tensor[] values)
        {
            output = values[0];
            value_index = values[1];
        }

        public Tensor this[int idx]
        {
            get
            {
                switch (idx)
                {
                    case 0:
                        return output;
                    case 1:
                        return value_index;
                    default:
                        return null;
                }
            }
        }

        public static implicit operator Tensor(MergeOutput merge)
            => merge.output;
    }
}
