using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Training.Saving.SavedModel
{
    /// <summary>
    /// A context for building a graph of SavedModel.
    /// </summary>
    public static class SaveContext
    {
        // TODO: make it thead safe.
        private static bool _in_save_context = false;
        private static SaveOptions _save_options = null;

        public static bool in_save_context() => _in_save_context;
        public static SaveOptions get_save_options()
        {
            if (!in_save_context())
            {
                throw new ValueError("Not in a SaveContext.");
            }
            return _save_options;
        }
        public static SaveContextHandler save_context(SaveOptions options)
        {
            return new SaveContextHandler(options);
        }
        
        public class SaveContextHandler: IDisposable
        {
            private bool _old_in_save_context;
            private SaveOptions _old_save_options;
            public SaveContextHandler(SaveOptions options)
            {
                if (SaveContext.in_save_context())
                {
                    throw new ValueError("Already in a SaveContext.");
                }
                _old_in_save_context = SaveContext._in_save_context;
                SaveContext._in_save_context = true;
                _old_save_options = SaveContext._save_options;
                SaveContext._save_options = options;
            }
            public void Dispose()
            {
                SaveContext._in_save_context = _old_in_save_context;
                SaveContext._save_options = _old_save_options;
            }
        }
    }
}
