namespace Tensorflow.Training
{
    /// <summary>
    /// Base timer for determining when Hooks should trigger.
    /// </summary>
    public abstract class _HookTimer
    {
        /// <summary>
        /// Resets the timer.
        /// </summary>
        public abstract void reset();

        /// <summary>
        /// Return true if the timer should trigger for the specified step.
        /// </summary>
        /// <param name="step"></param>
        /// <returns></returns>
        public abstract bool should_trigger_for_step(int step);

        /// <summary>
        /// Update the last triggered time and step number.
        /// </summary>
        /// <param name="step"></param>
        public abstract void update_last_triggered_step(int step);

        /// <summary>
        /// Returns the last triggered time step or None if never triggered.
        /// </summary>
        /// <returns></returns>
        public abstract int last_triggered_step();
    }
}
