namespace Tensorflow
{
    public class Reduction
    {
        public const string NONE = "none";
        public const string SUM = "sum";
        public const string WEIGHTED_SUM = "weighted_sum";
        public const string SUM_OVER_BATCH_SIZE = "weighted_sum_over_batch_size";
        public const string WEIGHTED_MEAN = "weighted_mean";
        public const string SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights";
        public const string SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS;
    }
}
