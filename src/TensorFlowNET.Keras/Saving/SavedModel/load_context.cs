using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Tensorflow.Training.Saving.SavedModel;

namespace Tensorflow.Keras.Saving.SavedModel
{
    // TODO: remove this class to common project.
    public class ContextHandler: IDisposable
    {
        public Action<bool> DisposeCallBack { get; set; }
        public void Dispose()
        {
            DisposeCallBack.Invoke(true);
        }
    }
    public class LoadContext
    {
        private bool _entered_load_context;
        private LoadOptions? _load_options;
        private static ThreadLocal<LoadContext> _load_context = new();
        private LoadContext()
        {
            _entered_load_context = false;
            _load_options = null;
        }

        public void set_load_options(LoadOptions load_options)
        {
            _load_options = load_options;
            _entered_load_context = true;
        }

        private void clear_load_options()
        {
            _load_options = null;
            _entered_load_context = false;
        }

        private LoadOptions? load_options()
        {
            return _load_options;
        }

        public static ContextHandler load_context(LoadOptions? load_options)
        {
            if(_load_context.Value is null)
            {
                _load_context.Value = new LoadContext();
            }
            _load_context.Value.set_load_options(load_options);
            return new ContextHandler()
            {
                DisposeCallBack = _ => _load_context.Value.clear_load_options()
            };
        }

        public static LoadOptions? get_load_option()
        {
            return _load_context.Value.load_options();
        }

        public static bool in_load_context()
        {
            return _load_context.Value._entered_load_context;
        }
    }
}
