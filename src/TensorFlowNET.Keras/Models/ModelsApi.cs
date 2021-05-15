using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using ThirdParty.Tensorflow.Python.Keras.Protobuf;

namespace Tensorflow.Keras.Models
{
    public class ModelsApi
    {
        public Functional from_config(ModelConfig config)
            => Functional.from_config(config);

        public void load_model(string filepath, bool compile = true)
        {
            var bytes = File.ReadAllBytes(Path.Combine(filepath, "saved_model.pb"));
            var saved_mode = SavedModel.Parser.ParseFrom(bytes);
            
            var meta_graph_def = saved_mode.MetaGraphs[0];
            var object_graph_def = meta_graph_def.ObjectGraphDef;

            bytes = File.ReadAllBytes(Path.Combine(filepath, "keras_metadata.pb"));
            var metadata = SavedMetadata.Parser.ParseFrom(bytes);

            // Recreate layers and metrics using the info stored in the metadata.
            var keras_loader = new KerasObjectLoader(metadata, object_graph_def);
            keras_loader.load_layers(compile: compile);
        }
    }
}
