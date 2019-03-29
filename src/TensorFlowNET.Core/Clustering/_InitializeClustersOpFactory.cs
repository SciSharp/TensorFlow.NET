using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Clustering
{
    /// <summary>
    /// Internal class to create the op to initialize the clusters.
    /// </summary>
    public class _InitializeClustersOpFactory : Python
    {
        Tensor[] _inputs;
        Tensor _num_clusters;
        string _initial_clusters;
        string _distance_metric;
        int _random_seed;
        int _kmeans_plus_plus_num_retries;
        int _kmc2_chain_length;
        RefVariable _cluster_centers;
        RefVariable _cluster_centers_updated;
        RefVariable _cluster_centers_initialized;
        Tensor _num_selected;
        Tensor _num_remaining;
        Tensor _num_data;

        public _InitializeClustersOpFactory(Tensor[] inputs,
            Tensor num_clusters,
            string initial_clusters,
            string distance_metric,
            int random_seed,
            int kmeans_plus_plus_num_retries,
            int kmc2_chain_length,
            RefVariable cluster_centers,
            RefVariable cluster_centers_updated,
            RefVariable cluster_centers_initialized)
        {
            _inputs = inputs;
            _num_clusters = num_clusters;
            _initial_clusters = initial_clusters;
            _distance_metric = distance_metric;
            _random_seed = random_seed;
            _kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries;
            _kmc2_chain_length = kmc2_chain_length;
            _cluster_centers = cluster_centers;
            _cluster_centers_updated = cluster_centers_updated;
            _cluster_centers_initialized = cluster_centers_initialized;

            _num_selected = array_ops.shape(_cluster_centers)[0];
            _num_remaining = _num_clusters - _num_selected;

            _num_data = math_ops.add_n(_inputs.Select(i => array_ops.shape(i)[0]).ToArray());
        }

        public Tensor[] op()
        {
            return control_flow_ops.cond(gen_math_ops.equal(_num_remaining, 0),
                () => new Operation[] { check_ops.assert_equal(_cluster_centers_initialized, true) },
                _initialize);
        }

        private Operation[] _initialize()
        {
            with(ops.control_dependencies(new Operation[]
            {
                check_ops.assert_positive(_num_remaining)
            }), delegate
            {
                // var num_now_remaining = _add_new_centers();
            });

            throw new NotImplementedException("_InitializeClustersOpFactory _initialize");
        }

        /*private int _add_new_centers()
        {
            var new_centers = _choose_initial_centers();
        }*/
    }
}
