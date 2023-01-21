namespace Tensorflow.Train
{
    public abstract class AutoTrackable : Trackable
    {
        public void _delete_tracking(string name)
        {
            _maybe_initialize_trackable();
            if (_unconditional_dependency_names.ContainsKey(name))
            {
                _unconditional_dependency_names.Remove(name);
                for (int i = _unconditional_checkpoint_dependencies.Count - 1; i >= 0; i--)
                {
                    if (_unconditional_checkpoint_dependencies[i].Name == name)
                    {
                        _unconditional_checkpoint_dependencies.RemoveAt(i);
                    }
                }
            }
        }
    }
}
