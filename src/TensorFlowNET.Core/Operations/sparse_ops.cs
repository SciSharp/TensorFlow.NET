namespace Tensorflow
{
    public class sparse_ops
    {
        /// <summary>
        /// Converts a sparse representation into a dense tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sparse_indices"></param>
        /// <param name="output_shape"></param>
        /// <param name="sparse_values"></param>
        /// <param name="default_value"></param>
        /// <param name="validate_indices"></param>
        /// <param name="name"></param>
        /// <returns>Dense `Tensor` of shape `output_shape`.  Has the same type as `sparse_values`.</returns>
        public Tensor sparse_to_dense<T>(Tensor sparse_indices,
            int[] output_shape,
            T sparse_values,
            T default_value = default,
            bool validate_indices = true,
            string name = null)
            => gen_sparse_ops.sparse_to_dense(sparse_indices,
                output_shape,
                sparse_values,
                default_value: default_value,
                validate_indices: validate_indices,
                name: name);
    }
}
