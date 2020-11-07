namespace Tensorflow.Gradients
{
    public class AccumulatorCallState
    {
        GradientTape backward_tape;
        bool accumulating;

        public AccumulatorCallState(GradientTape backward_tape, bool accumulating)
        {
            this.backward_tape = backward_tape;
            this.accumulating = accumulating;
        }
    }
}
