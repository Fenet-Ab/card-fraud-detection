import numpy as np


class QLearningAgent:

    def __init__(
        self,
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):

        self.n_actions = n_actions

        # Learning rate
        self.learning_rate = learning_rate

        # Importance of future rewards
        self.discount_factor = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # Exploration decay
        self.epsilon_decay = epsilon_decay

        # Minimum exploration
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = {}

    # ============================================================
    # STATE DISCRETIZATION
    # ============================================================

    def _state_to_key(self, state):

        probability = float(state[0])
        amount = float(state[1])
        budget = float(state[2])

        # --------------------------------------------------------
        # Fraud probability bins
        # --------------------------------------------------------

        probability_bins = [
            0.001,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75
        ]

        probability_state = np.digitize(
            probability,
            probability_bins
        )

        # --------------------------------------------------------
        # Transaction amount bins
        # --------------------------------------------------------

        amount_bins = [
            0.10,
            0.25,
            0.50,
            0.75
        ]

        amount_state = np.digitize(
            amount,
            amount_bins
        )

        # --------------------------------------------------------
        # Alert budget bins
        # --------------------------------------------------------

        budget_bins = [
            0.10,
            0.25,
            0.50,
            0.75
        ]

        budget_state = np.digitize(
            budget,
            budget_bins
        )

        return (
            int(probability_state),
            int(amount_state),
            int(budget_state)
        )

    # ============================================================
    # GET Q VALUES
    # ============================================================

    def get_q_values(self, state):

        key = self._state_to_key(state)

        if key not in self.q_table:

            self.q_table[key] = np.zeros(
                self.n_actions
            )

        return self.q_table[key]

    # ============================================================
    # CHOOSE ACTION
    # ============================================================

    def choose_action(self, state):

        q_values = self.get_q_values(state)

        # --------------------------------------------------------
        # Exploration
        # --------------------------------------------------------

        if np.random.random() < self.epsilon:

            return np.random.randint(
                self.n_actions
            )

        # --------------------------------------------------------
        # Exploitation
        # --------------------------------------------------------

        return int(
            np.argmax(q_values)
        )

    # ============================================================
    # UPDATE Q VALUE
    # ============================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        q_values = self.get_q_values(
            state
        )

        current_q = q_values[action]

        # --------------------------------------------------------
        # Terminal state
        # --------------------------------------------------------

        if done:

            target = reward

        else:

            next_q_values = self.get_q_values(
                next_state
            )

            target = (
                reward
                + self.discount_factor
                * np.max(next_q_values)
            )

        # --------------------------------------------------------
        # Q-learning update
        # --------------------------------------------------------

        new_q = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

        q_values[action] = new_q

    # ============================================================
    # DECAY EXPLORATION
    # ============================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )