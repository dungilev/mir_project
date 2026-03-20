from enum import Enum, auto


class RobotState(Enum):
    IDLE = auto()
    GOING_TO_TABLE = auto()
    TAKING_ORDER = auto()
    CONFIRMING_ORDER = auto()
    GOING_TO_KITCHEN = auto()
    WAITING_FOR_FOOD = auto()
    DELIVERING_FOOD = auto()
    SERVING_CUSTOMER = auto()
    RETURNING_HOME = auto()
    PROCESSING_PAYMENT = auto()


class StateMachine:
    def __init__(self):
        self.state = RobotState.IDLE
        self.current_table = None

    def get_state(self):
        return self.state

    def transition_to(self, state):
        self.state = state

    def get_state_name(self):
        return self.state.name

    def get_state_description(self):
        return "Sẵn sàng nhận lệnh"
