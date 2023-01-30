using System;
using System.Collections.Generic;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel
{
    public interface ISerializedAttributes
    {
        IDictionary<string, Trackable> Functions { get; }

        IDictionary<string, Trackable> CheckpointableObjects { get; }

        /// <summary>
        /// Returns functions to attach to the root object during serialization.
        /// </summary>
        IDictionary<string, Trackable> FunctionsToSerialize { get; }

        /// <summary>
        /// Returns objects to attach to the root object during serialization.
        /// </summary>
        IDictionary<string, Trackable> ObjectsToSerialize{get; }

        /// <summary>
        /// Saves function dictionary, and validates dictionary values.
        /// </summary>
        /// <param name="function_dict"></param>
        IDictionary<string, Trackable> set_and_validate_functions(IDictionary<string, Trackable> function_dict);

        /// <summary>
        /// Saves objects to a dictionary, and validates the values.
        /// </summary>
        /// <param name="object_dict"></param>
        IDictionary<string, Trackable> set_and_validate_objects(IDictionary<string, Trackable> object_dict);
    }
}
