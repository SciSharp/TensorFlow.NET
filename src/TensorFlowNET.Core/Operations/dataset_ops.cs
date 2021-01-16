using System;
using Tensorflow.Framework.Models;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class dataset_ops
    {
        public Tensor tensor_dataset(Tensor[] components, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "TensorDataset", name,
                    null,
                    new object[]
                    {
                        components,
                        "output_shapes", output_shapes
                    });
                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("TensorDataset",
                name: name,
                args: new { components, output_shapes });

            return _op.output;
        }

        /// <summary>
        /// Creates a dataset that emits each dim-0 slice of `components` once.
        /// </summary>
        /// <param name="components"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor tensor_slice_dataset(Tensor[] components, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "TensorSliceDataset", name,
                    null,
                    new object[]
                    {
                        components,
                        "output_shapes", output_shapes
                    });
                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("TensorSliceDataset",
                name: name,
                args: new { components, output_shapes });

            return _op.outputs[0];
        }

        public Tensor range_dataset(Tensor start, Tensor stop, Tensor step, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "RangeDataset", name,
                    null,
                    start, stop, step,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor repeat_dataset(Tensor input_dataset, Tensor count, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "RepeatDataset", name,
                    null,
                    input_dataset, count,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor shard_dataset(Tensor input_dataset, Tensor num_shards, Tensor index,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool require_non_empty = false, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ShardDataset", name,
                    null,
                    input_dataset, num_shards, index,
                    "require_non_empty", require_non_empty,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor zip_dataset(Tensor[] input_datasets,
            TF_DataType[] output_types,
            TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ZipDataset", name,
                    null,
                    new object[]
                    {
                        input_datasets,
                        "output_types", output_types,
                        "output_shapes", output_shapes
                    });
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor shuffle_dataset_v3(Tensor input_dataset, Tensor buffer_size,
            Tensor seed, Tensor seed2, Tensor seed_generator,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool reshuffle_each_iteration = true,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ShuffleDatasetV3", name,
                    null,
                    input_dataset, buffer_size,
                    seed, seed2, seed_generator,
                    "reshuffle_each_iteration", reshuffle_each_iteration,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor skip_dataset(Tensor input_dataset, Tensor count,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "SkipDataset", name,
                    null,
                    input_dataset, count,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor dummy_seed_generator(string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "DummySeedGenerator", name,
                    null);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public Tensor concatenate_dataset(Tensor input_dataset, Tensor another_dataset,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ConcatenateDataset", name,
                    null,
                    input_dataset, another_dataset,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("ConcatenateDataset",
                name: name,
                args: new { input_dataset, another_dataset, output_types, output_shapes });

            return _op.outputs[0];
        }

        public Tensor cache_dataset_v2(Tensor input_dataset, Tensor filename, Tensor cache,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "CacheDatasetV2", name,
                    null,
                    input_dataset, filename, cache,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset that batches `batch_size` elements from `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="buffer_size"></param>
        /// <param name="drop_remainder"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="parallel_copy"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor batch_dataset_v2(Tensor input_dataset, Tensor buffer_size,
            Tensor drop_remainder,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool parallel_copy = false,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "BatchDatasetV2", name,
                    null,
                    input_dataset, buffer_size, drop_remainder,
                    "parallel_copy", parallel_copy,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor dummy_memory_cache(string name = "")
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "DummyMemoryCache", name,
                    null);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset that asynchronously prefetches elements from `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="buffer_size"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="slack_period"></param>
        /// <param name="legacy_autotune"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor prefetch_dataset(Tensor input_dataset, Tensor buffer_size,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            int? slack_period = 0,
            bool legacy_autotune = true,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "PrefetchDataset", name,
                    null,
                    input_dataset, buffer_size,
                    "output_types", output_types,
                    "output_shapes", output_shapes,
                    "slack_period", slack_period,
                    "legacy_autotune", legacy_autotune);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset that contains `count` elements from the `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="count"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor take_dataset(Tensor input_dataset, Tensor count,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "TakeDataset", name,
                    null,
                    input_dataset, count,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset by applying optimizations to `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="optimizations"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="optimization_configs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor optimize_dataset(Tensor input_dataset, Tensor optimizations,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string[] optimization_configs = null,
            string name = null)
        {
            if (optimization_configs == null)
                optimization_configs = new string[0];

            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "OptimizeDataset", name,
                    null,
                    input_dataset, optimizations,
                    "output_types", output_types,
                    "output_shapes", output_shapes,
                    "optimization_configs", optimization_configs);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Identity transformation that models performance.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="algorithm"></param>
        /// <param name="cpu_budget"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor model_dataset(Tensor input_dataset,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            AutotuneAlgorithm algorithm, long cpu_budget,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ModelDataset", name,
                    null,
                    input_dataset,
                    "algorithm", algorithm,
                    "cpu_budget", cpu_budget,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// A container for an iterator resource.
        /// </summary>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns>A tuple of `Tensor` objects (handle, deleter).</returns>
        public (Tensor, Tensor) anonymous_iterator_v2(TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "AnonymousIteratorV2", name,
                    null,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return (results[0], results[1]);
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Makes a new iterator from the given `dataset` and stores it in `iterator`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="iterator"></param>
        /// <param name="name"></param>
        /// <returns>The created Operation.</returns>
        public ITensorOrOperation make_iterator(Tensor dataset, Tensor iterator, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MakeIterator", name,
                    null,
                    dataset, iterator);
                return null;
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="iterator"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor map_dataset(Tensor dataset, ConcreteFunction f, TF_DataType[] output_types, TensorShape[] output_shapes,
            bool use_inter_op_parallelism = true, bool preserve_cardinality = false, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MapDataset", name,
                    null,
                    dataset, new Tensor[0],
                    "f", f,
                    "output_types", output_types,
                    "output_shapes", output_shapes,
                    "use_inter_op_parallelism", use_inter_op_parallelism,
                    "preserve_cardinality", preserve_cardinality);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="f"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor flat_map_dataset(Tensor dataset, ConcreteFunction f, TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "FlatMapDataset", name,
                    null,
                    dataset, new Tensor[0],
                    "f", f,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="num_parallel_calls"></param>
        /// <param name="f"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="use_inter_op_parallelism"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor parallel_map_dataset_v2(Tensor dataset, Tensor num_parallel_calls, ConcreteFunction f,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool use_inter_op_parallelism = true,
            string deterministic = "default",
            bool preserve_cardinality = false,
            string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ParallelMapDatasetV2", name,
                    null,
                    dataset, new Tensor[0], num_parallel_calls,
                    "f", f,
                    "output_types", output_types,
                    "output_shapes", output_shapes,
                    "use_inter_op_parallelism", use_inter_op_parallelism,
                    "deterministic", deterministic,
                    "preserve_cardinality", preserve_cardinality);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// A container for an iterator resource.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="deleter"></param>
        /// <param name="name"></param>
        /// <returns>The created Operation.</returns>
        public ITensorOrOperation delete_iterator(Tensor handle, Tensor deleter, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "DeleteIterator", name,
                    null,
                    handle, deleter);
                return null;
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Gets the next output from the given iterator .
        /// </summary>
        /// <param name="iterator"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor[] iterator_get_next(Tensor iterator, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "IteratorGetNext", name,
                    null,
                    iterator,
                    "output_types", output_types,
                    "output_shapes", output_shapes);
                return results;
            }

            throw new NotImplementedException("");
        }
    }
}
