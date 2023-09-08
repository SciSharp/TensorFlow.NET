using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Metrics;

public class metrics_utils
{
    public static Tensor accuracy(Tensor y_true, Tensor y_pred)
    {
        if (y_true.dtype != y_pred.dtype)
            y_pred = tf.cast(y_pred, y_true.dtype);
        return tf.cast(tf.equal(y_true, y_pred), keras.backend.floatx());
    }

    public static Tensor binary_matches(Tensor y_true, Tensor y_pred, float threshold = 0.5f)
    {
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype);
        return tf.cast(tf.equal(y_true, y_pred), keras.backend.floatx());
    }

    public static Tensor cosine_similarity(Tensor y_true, Tensor y_pred, Axis? axis = null)
    {
        y_true = tf.linalg.l2_normalize(y_true, axis: axis ?? -1);
        y_pred = tf.linalg.l2_normalize(y_pred, axis: axis ?? -1);
        return tf.reduce_sum(y_true * y_pred, axis: axis ?? -1);
    }

    public static Tensor hamming_loss_fn(Tensor y_true, Tensor y_pred, Tensor threshold, string mode)
    {
        if (threshold == null)
        {
            threshold = tf.reduce_max(y_pred, axis: -1, keepdims: true);
            // make sure [0, 0, 0] doesn't become [1, 1, 1]
            // Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12f);
        }
        else
        {
            y_pred = y_pred > threshold;
        }


        y_true = tf.cast(y_true, tf.int32);
        y_pred = tf.cast(y_pred, tf.int32);

        if (mode == "multiclass")
        {
            var nonzero = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis: -1), tf.float32);
            return 1.0 - nonzero;
        }
        else
        {
            var nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis: -1), tf.float32);
            return nonzero / y_true.shape[-1];
        }
    }
    
    /// <summary>
    /// Creates float Tensor, 1.0 for label-prediction match, 0.0 for mismatch.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <returns></returns>
    public static Tensor sparse_categorical_matches(Tensor y_true, Tensor y_pred)
    {
        var reshape_matches = false;
        var y_true_rank = y_true.shape.ndim;
        var y_pred_rank = y_pred.shape.ndim;
        var y_true_org_shape = tf.shape(y_true);

        if (y_true_rank > -1 && y_pred_rank > -1 && y_true.ndim == y_pred.ndim )
        {
            reshape_matches = true;
            y_true = tf.squeeze(y_true, new Shape(-1));
        }
        y_pred = tf.math.argmax(y_pred, axis: -1);
        y_pred = tf.cast(y_pred, y_true.dtype);
        var matches = tf.cast(
            tf.equal(y_true, y_pred),
            dtype: keras.backend.floatx()
        );

        if (reshape_matches)
        {
            return tf.reshape(matches, shape: y_true_org_shape);
        }

        return matches;
    }

    public static Tensor sparse_top_k_categorical_matches(Tensor y_true, Tensor y_pred, int k = 5)
    {
        var reshape_matches = false;
        var y_true_rank = y_true.shape.ndim;
        var y_pred_rank = y_pred.shape.ndim;
        var y_true_org_shape = tf.shape(y_true);

        if (y_pred_rank > 2)
        {
            y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]));
        }
            
        if (y_true_rank > 1)
        {
            reshape_matches = true;
            y_true = tf.reshape(y_true, new Shape(-1));
        }

        var matches = tf.cast(
            tf.math.in_top_k(
                predictions: y_pred, targets: tf.cast(y_true, np.int32), k: k
            ),
            dtype: keras.backend.floatx()
        );

        if (reshape_matches)
        {
            return tf.reshape(matches, shape: y_true_org_shape);
        }
        
        return matches;
    }

    public static Tensor update_confusion_matrix_variables(Dictionary<string, IVariableV1> variables_to_update,
        Tensor y_true,
        Tensor y_pred,
        Tensor thresholds,
        int top_k,
        int class_id,
        Tensor sample_weight = null,
        bool multi_label = false,
        Tensor label_weights = null,
        bool thresholds_distributed_evenly = false)
    {
        var variable_dtype = variables_to_update.Values.First().dtype;
        y_true = tf.cast(y_true, dtype: variable_dtype);
        y_pred = tf.cast(y_pred, dtype: variable_dtype);
        var num_thresholds = thresholds.shape.dims[0];

        Tensor one_thresh = null;
        if (multi_label)
        {
            one_thresh = tf.equal(tf.cast(constant_op.constant(1), dtype:tf.int32),
                tf.rank(thresholds),
                name: "one_set_of_thresholds_cond");
        }
        else
        {
            one_thresh = tf.cast(constant_op.constant(true), dtype: dtypes.@bool);
        }

        if (sample_weight == null)
        {
            (y_pred, y_true, _) = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true);
        }
        else
        {
            sample_weight = tf.cast(sample_weight, dtype: variable_dtype);
            (y_pred, y_true, sample_weight) = losses_utils.squeeze_or_expand_dimensions(y_pred, 
                y_true, 
                sample_weight: sample_weight);
        }

        if (top_k > 0)
        {
            y_pred = _filter_top_k(y_pred, top_k);
        }

        if (class_id > 0)
        {
            y_true = y_true[Slice.All, class_id];
            y_pred = y_pred[Slice.All, class_id];
        }

        if (thresholds_distributed_evenly)
        {
            throw new NotImplementedException();
        }

        var pred_shape = tf.shape(y_pred);
        var num_predictions = pred_shape[0];

        Tensor num_labels;
        if (y_pred.shape.ndim == 1)
        {
            num_labels = constant_op.constant(1);
        }
        else
        {
            num_labels = tf.reduce_prod(pred_shape["1:"], axis: 0);
        }
        var thresh_label_tile = tf.where(one_thresh, num_labels, tf.ones(new int[0], dtype: tf.int32));

        // Reshape predictions and labels, adding a dim for thresholding.
        Tensor predictions_extra_dim, labels_extra_dim;
        if (multi_label)
        {
            predictions_extra_dim = tf.expand_dims(y_pred, 0);
            labels_extra_dim = tf.expand_dims(tf.cast(y_true, dtype: tf.@bool), 0);
        }

        else
        {
            // Flatten predictions and labels when not multilabel.
            predictions_extra_dim = tf.reshape(y_pred, (1, -1));
            labels_extra_dim = tf.reshape(tf.cast(y_true, dtype: tf.@bool), (1, -1));
        }

        // Tile the thresholds for every prediction.
        object[] thresh_pretile_shape, thresh_tiles, data_tiles;
        
        if (multi_label)
        {
            thresh_pretile_shape = new object[] { num_thresholds, 1, -1 };
            thresh_tiles = new object[] { 1, num_predictions, thresh_label_tile };
            data_tiles = new object[] { num_thresholds, 1, 1 };
        }
        else
        {
            thresh_pretile_shape = new object[] { num_thresholds, -1 };
            thresh_tiles = new object[] { 1, num_predictions * num_labels };
            data_tiles = new object[] { num_thresholds, 1 };
        }
        var thresh_tiled = tf.tile(tf.reshape(thresholds, thresh_pretile_shape), tf.stack(thresh_tiles));

        // Tile the predictions for every threshold.
        var preds_tiled = tf.tile(predictions_extra_dim, data_tiles);

        // Compare predictions and threshold.
        var pred_is_pos = tf.greater(preds_tiled, thresh_tiled);

        // Tile labels by number of thresholds
        var label_is_pos = tf.tile(labels_extra_dim, data_tiles);

        Tensor weights_tiled = null;

        if (sample_weight != null)
        {
            /*sample_weight = broadcast_weights(
                tf.cast(sample_weight, dtype: variable_dtype), y_pred);*/
            weights_tiled = tf.tile(
                tf.reshape(sample_weight, thresh_tiles), data_tiles);
        }

        if (label_weights != null && !multi_label)
        {
            throw new NotImplementedException();
        }

        Func<Tensor, Tensor, Tensor, IVariableV1, ITensorOrOperation> weighted_assign_add
            = (label, pred, weights, var) =>
            {
                var label_and_pred = tf.cast(tf.logical_and(label, pred), dtype: var.dtype);
                if (weights != null)
                {
                    label_and_pred *= tf.cast(weights, dtype: var.dtype);
                }

                return var.assign_add(tf.reduce_sum(label_and_pred, 1));
            };


        var loop_vars = new Dictionary<string, (Tensor, Tensor)>
        {
            { "tp", (label_is_pos, pred_is_pos) }
        };
        var update_tn = variables_to_update.ContainsKey("tn");
        var update_fp = variables_to_update.ContainsKey("fp");
        var update_fn = variables_to_update.ContainsKey("fn");

        Tensor pred_is_neg = null;
        if (update_fn || update_tn)
        {
            pred_is_neg = tf.logical_not(pred_is_pos);
            loop_vars["fn"] = (label_is_pos, pred_is_neg);
        }

        if(update_fp || update_tn)
        {
            var label_is_neg = tf.logical_not(label_is_pos);
            loop_vars["fp"] = (label_is_neg, pred_is_pos);
            if (update_tn)
            {
                loop_vars["tn"] = (label_is_neg, pred_is_neg);
            }
        }

        var update_ops = new List<ITensorOrOperation>();
        foreach (var matrix_cond in loop_vars.Keys)
        {
            var (label, pred) = loop_vars[matrix_cond];
            if (variables_to_update.ContainsKey(matrix_cond))
            {
                var op = weighted_assign_add(label, pred, weights_tiled, variables_to_update[matrix_cond]);
                update_ops.append(op);
            }
        }

        tf.group(update_ops.ToArray());
        return null;
    }

    private static Tensor _filter_top_k(Tensor x, int k)
    {
        var NEG_INF = -1e10;
        var (_, top_k_idx) = tf.math.top_k(x, k, sorted: false);
        var top_k_mask = tf.reduce_sum(
            tf.one_hot(top_k_idx.Single, (int)x.shape[-1], axis: -1), axis: -2);
        return x * top_k_mask + NEG_INF * (1 - top_k_mask);
    }
}
