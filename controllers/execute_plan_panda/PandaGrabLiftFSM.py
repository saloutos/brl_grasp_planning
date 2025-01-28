# imports
from controllers.base.FiniteStateMachine import *
from .states.reset import *
from .states.planning import *
from .states.move_to_approach import *
from .states.move_to_pregrasp import *
from .states.execute_grasp import *
from .states.lift_object import *
from .states.hold_object import *
from .states.release_object import *

# define reflex FSM class
class PandaGrabLiftFSM(FiniteStateMachine):
    def __init__(self, states=[]):
        states = [Reset(),
                Planning(),
                MoveToApproach(),
                MoveToPreGrasp(),
                ExecuteGrasp(),
                LiftObject(),
                HoldObject(),
                ReleaseObject()] # Note: FSM will start in first state in states list above
        self.states = dict(zip([state.name for state in states], states))
        self.current_state = ""
        self.current_state_idx = 0
        self.next_state = states[0].name # the first state in the list is the starting state

    def begin(self, GP):
        # TODO: add any other initialization code here?
        self.states[self.next_state].enter(GP)
        self.current_state = self.next_state
        self.current_state_idx = list(self.states.keys()).index(self.current_state)

    def update(self, GP):

        # execute FSM
        self.next_state = self.states[self.current_state].execute(GP) # check for user inputs within functions

        # if there was a valid state transition, update state
        if (self.next_state != self.current_state):
            if (self.next_state in self.states.keys()):
                # handle transition
                # TODO: add better error handling here
                self.states[self.current_state].exit(GP)
                self.states[self.next_state].enter(GP)
                # reset state tracking
                self.current_state = self.next_state
                self.current_state_idx = list(self.states.keys()).index(self.current_state)
            else:
                # TODO: need to test this
                print("Attempted a bad state transition. " + self.next_state + " is not a valid state name!")