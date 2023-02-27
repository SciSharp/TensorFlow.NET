using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Train;
using Tensorflow.Training.Saving.SavedModel;

namespace Tensorflow.ModelSaving
{
    public class ModelSaver
    {
        public void save(Trackable obj, string export_dir, SaveOptions options = null)
        {
            var saved_model = new SavedModel();
            var meta_graph_def = new MetaGraphDef();
            saved_model.MetaGraphs.Add(meta_graph_def);
            _build_meta_graph(obj, export_dir, options, meta_graph_def);
        }

        void _build_meta_graph(Trackable obj, string export_dir, SaveOptions options,
            MetaGraphDef meta_graph_def = null)
        {

        }
    }
}
