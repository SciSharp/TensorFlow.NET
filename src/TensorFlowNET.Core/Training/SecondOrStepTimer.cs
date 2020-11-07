using System;

namespace Tensorflow.Training
{
    public class SecondOrStepTimer : _HookTimer
    {
        int _every_secs = 60;
        int _every_steps = 0;
        int _last_triggered_step = 0;
#pragma warning disable CS0414 // The field 'SecondOrStepTimer._last_triggered_time' is assigned but its value is never used
        int _last_triggered_time = 0;
#pragma warning restore CS0414 // The field 'SecondOrStepTimer._last_triggered_time' is assigned but its value is never used

        public SecondOrStepTimer(int every_secs, int every_steps)
        {
            _every_secs = every_secs;
            _every_steps = every_steps;
        }

        public override void reset()
        {
            _last_triggered_step = 0;
            _last_triggered_time = 0;
        }

        public override int last_triggered_step()
        {
            return _last_triggered_step;
        }

        public override bool should_trigger_for_step(int step)
        {
            throw new NotImplementedException();
        }

        public override void update_last_triggered_step(int step)
        {
            throw new NotImplementedException();
        }
    }
}
