def calculate_reward(actual_class, action):
    """
    Calculate the reward for an RL action.

    actual_class:
        0 = legitimate
        1 = fraud

    action:
        0 = approve
        1 = review
        2 = block
    """

    # Legitimate transaction
    if actual_class == 0:

        if action == 0:       # APPROVE
            return 1

        elif action == 1:     # REVIEW
            return -1

        elif action == 2:     # BLOCK
            return -5

    # Fraudulent transaction
    elif actual_class == 1:

        if action == 0:       # APPROVE
            return -10

        elif action == 1:     # REVIEW
            return 5

        elif action == 2:     # BLOCK
            return 10

    raise ValueError("Invalid class or action")