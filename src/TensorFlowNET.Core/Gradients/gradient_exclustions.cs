namespace Tensorflow.Gradients
{
    public class gradient_exclustions
    {
        public static int[] OpGradientUnusedInputIndices(string op_name)
            => op_name switch
            {
                "FusedBatchNorm" => new[] { 2 },
                "FusedBatchNormGradV3" => new[] { 5 },
                "FusedBatchNormV2" => new[] { 2 },
                "FusedBatchNormV3" => new[] { 2 },
                "ReadVariableOp" => new int[0],
                _ => null
            };

        public static int[] OpGradientUnusedOutputIndices(string op_name)
            => op_name switch
            {
                "FusedBatchNormV3" => new[] { 0, 1, 2 },
                "ReadVariableOp" => new int[0],
                "SoftmaxCrossEntropyWithLogits" => new[] { 0 },
                "TensorArrayConcat" => new[] { 0 },
                "TensorArrayConcatV2" => new[] { 0 },
                "TensorArrayConcatV3" => new[] { 0 },
                "Mul" => new int[0],
                "Sum" => new int[0],
                _ => null
            };
    }
}
