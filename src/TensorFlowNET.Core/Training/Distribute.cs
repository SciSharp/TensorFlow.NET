namespace Tensorflow
{
    public static class Distribute
    {
        public static VariableAggregationType get_loss_reduction()
        {
            return VariableAggregationType.MEAN;
        }
    }
}
