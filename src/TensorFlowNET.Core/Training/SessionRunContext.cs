namespace Tensorflow.Training
{
    public class SessionRunContext
    {
        SessionRunArgs _original_args;
        public SessionRunArgs original_args => _original_args;

        Session _session;
        public Session session => _session;

        bool _stop_requested;
        public bool stop_requested => _stop_requested;

        public SessionRunContext(SessionRunArgs original_args, Session session)
        {
            _original_args = original_args;
            _session = session;
            _stop_requested = false;
        }

        public void request_stop()
        {
            _stop_requested = true;
        }
    }
}
