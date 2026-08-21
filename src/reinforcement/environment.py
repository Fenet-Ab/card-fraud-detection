import numpy as np

from src.reinforcement.reward import calculate_reward


class FraudDetectionEnvironment:

    # ============================================================
    # ACTIONS
    # ============================================================

    APPROVE = 0
    REVIEW = 1
    BLOCK = 2

    def __init__(
        self,
        probabilities,
        amounts,
        actual_labels,
        alert_budget=500
    ):

        # --------------------------------------------------------
        # Store transaction data
        # --------------------------------------------------------

        self.probabilities = np.asarray(
            probabilities
        )

        self.amounts = np.asarray(
            amounts
        )

        self.actual_labels = np.asarray(
            actual_labels
        )

        # --------------------------------------------------------
        # Validate input lengths
        # --------------------------------------------------------

        if not (
            len(self.probabilities)
            == len(self.amounts)
            == len(self.actual_labels)
        ):
            raise ValueError(
                "probabilities, amounts, and actual_labels "
                "must have the same length."
            )

        # --------------------------------------------------------
        # Alert budget
        # --------------------------------------------------------

        self.initial_alert_budget = alert_budget

        self.alert_budget = alert_budget

        # --------------------------------------------------------
        # Dataset information
        # --------------------------------------------------------

        self.n_transactions = len(
            self.actual_labels
        )

        # Current position inside the episode
        self.current_index = 0

        # Transaction indices used by the current episode
        self.indices = np.arange(
            self.n_transactions
        )

        # Number of transactions in current episode
        self.episode_size = self.n_transactions

    # ============================================================
    # RESET ENVIRONMENT
    # ============================================================

    def reset(
        self,
        shuffle=True,
        episode_size=None
    ):

        # --------------------------------------------------------
        # Reset position
        # --------------------------------------------------------

        self.current_index = 0

        # --------------------------------------------------------
        # Reset alert budget
        # --------------------------------------------------------

        self.alert_budget = (
            self.initial_alert_budget
        )

        # --------------------------------------------------------
        # Create transaction order
        # --------------------------------------------------------

        if shuffle:

            indices = np.random.permutation(
                self.n_transactions
            )

        else:

            indices = np.arange(
                self.n_transactions
            )

        # --------------------------------------------------------
        # Limit episode size
        # --------------------------------------------------------

        if episode_size is not None:

            if episode_size <= 0:

                raise ValueError(
                    "episode_size must be greater than 0."
                )

            episode_size = min(
                episode_size,
                self.n_transactions
            )

            indices = indices[
                :episode_size
            ]

        # --------------------------------------------------------
        # Save transaction indices
        # --------------------------------------------------------

        self.indices = indices

        self.episode_size = len(
            self.indices
        )

        # --------------------------------------------------------
        # Return initial state
        # --------------------------------------------------------

        return self._get_state()

    # ============================================================
    # GET CURRENT STATE
    # ============================================================

    def _get_state(self):

        # --------------------------------------------------------
        # Episode finished
        # --------------------------------------------------------

        if (
            self.current_index
            >= self.episode_size
        ):

            return None

        # --------------------------------------------------------
        # Get actual dataset index
        #
        # current_index = position inside episode
        # real_index    = position inside original dataset
        # --------------------------------------------------------

        real_index = self.indices[
            self.current_index
        ]

        # --------------------------------------------------------
        # Get transaction information
        # --------------------------------------------------------

        probability = self.probabilities[
            real_index
        ]

        amount = self.amounts[
            real_index
        ]

        # ========================================================
        # NORMALIZE FRAUD PROBABILITY
        # ========================================================

        # Fraud probability is already between 0 and 1.

        probability_normalized = np.clip(
            probability,
            0.0,
            1.0
        )

        # ========================================================
        # NORMALIZE TRANSACTION AMOUNT
        # ========================================================

        # Log transformation reduces the effect
        # of extremely large transactions.

        amount_log = np.log1p(
            max(float(amount), 0.0)
        )

        # Scale approximately to 0-1.

        amount_normalized = np.clip(
            amount_log / 10.0,
            0.0,
            1.0
        )

        # ========================================================
        # NORMALIZE ALERT BUDGET
        # ========================================================

        if self.initial_alert_budget > 0:

            budget_normalized = (
                self.alert_budget
                / self.initial_alert_budget
            )

        else:

            budget_normalized = 0.0

        budget_normalized = np.clip(
            budget_normalized,
            0.0,
            1.0
        )

        # ========================================================
        # CREATE STATE
        # ========================================================

        state = np.array(
            [
                probability_normalized,
                amount_normalized,
                budget_normalized
            ],
            dtype=np.float32
        )

        return state

    # ============================================================
    # TAKE ACTION
    # ============================================================

    def step(self, action):

        # --------------------------------------------------------
        # Check episode status
        # --------------------------------------------------------

        if (
            self.current_index
            >= self.episode_size
        ):

            raise RuntimeError(
                "Episode has already finished."
            )

        # --------------------------------------------------------
        # Validate action
        # --------------------------------------------------------

        if action not in (
            self.APPROVE,
            self.REVIEW,
            self.BLOCK
        ):

            raise ValueError(
                f"Invalid action: {action}. "
                f"Expected 0, 1, or 2."
            )

        # --------------------------------------------------------
        # Get original transaction index
        # --------------------------------------------------------

        real_index = self.indices[
            self.current_index
        ]

        # --------------------------------------------------------
        # Get actual transaction class
        # --------------------------------------------------------

        actual_class = int(
            self.actual_labels[
                real_index
            ]
        )

        # ========================================================
        # CALCULATE REWARD
        # ========================================================

        # REVIEW consumes one alert.

        if action == self.REVIEW:

            if self.alert_budget <= 0:

                # No alerts remaining.
                reward = -10

            else:

                # Consume one alert.

                self.alert_budget -= 1

                reward = calculate_reward(
                    actual_class,
                    action
                )

        else:

            reward = calculate_reward(
                actual_class,
                action
            )

        # --------------------------------------------------------
        # Move to next transaction
        # --------------------------------------------------------

        self.current_index += 1

        # --------------------------------------------------------
        # Check whether episode is finished
        # --------------------------------------------------------

        done = (
            self.current_index
            >= self.episode_size
        )

        # --------------------------------------------------------
        # Get next state
        # --------------------------------------------------------

        next_state = self._get_state()

        # --------------------------------------------------------
        # Information for monitoring
        # --------------------------------------------------------

        info = {
            "real_index": int(real_index),
            "actual_class": actual_class,
            "action": action,
            "alert_budget": self.alert_budget
        }

        # --------------------------------------------------------
        # Return environment result
        # --------------------------------------------------------

        return (
            next_state,
            reward,
            done,
            info
        )