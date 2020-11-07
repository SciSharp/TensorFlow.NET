namespace Tensorflow.Training
{
    /// <summary>
    /// Hook to extend calls to MonitoredSession.run().
    /// </summary>
    public abstract class SessionRunHook
    {
        /// <summary>
        /// Called once before using the session.
        /// </summary>
        public virtual void begin()
        {
        }

        /// <summary>
        /// Called when new TensorFlow session is created.
        /// </summary>
        /// <param name="session"></param>
        /// <param name="coord"></param>
        public virtual void after_create_session(Session session, Coordinator coord)
        {
        }

        /// <summary>
        /// Called before each call to run().
        /// </summary>
        /// <param name="run_context"></param>
        public virtual void before_run(SessionRunContext run_context)
        {
        }

        /// <summary>
        /// Called after each call to run().
        /// </summary>
        public virtual void after_run(SessionRunContext run_context, SessionRunValues run_values)
        {
        }

        /// <summary>
        /// Called at the end of session.
        /// </summary>
        public virtual void end(Session session)
        {
        }
    }
}
