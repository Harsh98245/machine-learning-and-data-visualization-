#harsh khandelwal
import numpy as np
import math

from station_state import StationState
from utilities import poisson_function

class YorkBikeRentalProblem:

    def __init__(self, max_bikes:int, lambda_0: int, lambda_1: int):
        """ Constructor for the York Bike Rental company """
        # build rentals and returns distributions for each location
        self.station_prob_tables = [[], []] #two bike stations
        self.station_lambdas = [lambda_0, lambda_1]
        self.curr_policy = {} #no trained policy yet
        self.max_bikes = max_bikes #max bikes at each station
        self.construct_prob_table(0)
        self.construct_prob_table(1)

    def construct_prob_table(self, station_id: int):
        """
        Returns p(num_rentals, num_returns | initial_bikes) for a given
        station with the input id (station_id) as NUMPY ARRAY:
            p[initial_bikes, num_rentals, num_returns]

        Note that you can assume num_rentals and num_returns at a station are
        INDEPENDENT given initial_bikes, and that both
        P(num_rentals|initial_bikes) and P(num_returns|initial_bikes)
        are drawn from POISSION DISTRIBUTIONS
        (with lambdas determined by the value of station_lambdas
        at the station with the correct input ID).
        This will simplify your calculations.

        The probability table you return should be a NUMPY ARRAY of size
        (max_bikes + 1) by (max_bikes + 1) by (max_bikes + 1)

        There should be a different value in the table for every combination of
        initial_bikes, num_rentals and num_returns and each value in the table
        should correspond to p(num_rentals, num_returns | initial_bikes).

        >>> a = YorkBikeRentalProblem(10, 3, 3) #max bikes 10, each station lambda is 3
        >>> a.construct_prob_table(0) #create prob tables for station 0
        >>> a.construct_prob_table(1) #create prob tables for station 1
        >>> len(a.station_prob_tables[0]) #max bikes 10? num entries is therefore 11.
        11
        >>> len(a.station_prob_tables[0][0])
        11
        >>> len(a.station_prob_tables[0][0][0])
        11
        >>> abs(sum(a.station_prob_tables[0][0][0].tolist()) - 1.0) < 0.00001
        True
        """
        lam = self.station_lambdas[station_id]
        n = self.max_bikes
        # Precompute poisson distributions for rentals and returns
        # P(k | lambda) for k in 0..n
        poisson = poisson_function(lam, n)

        # table[initial_bikes, num_rentals, num_returns]
        table = np.zeros((n + 1, n + 1, n + 1))

        for init in range(n + 1):
            for rentals in range(n + 1):
                # actual rentals capped by available bikes
                actual_rentals = min(rentals, init)
                p_rentals = poisson[rentals]
                for returns in range(n + 1):
                    p_returns = poisson[returns]
                    # Only contribute to the actual_rentals slot
                    if rentals == actual_rentals:
                        table[init, rentals, returns] = p_rentals * p_returns
                    else:
                        # lump probability of rentals > init into actual_rentals=init
                        pass

        # Re-build properly: for each init, rentals are capped at init
        table = np.zeros((n + 1, n + 1, n + 1))
        for init in range(n + 1):
            for returns in range(n + 1):
                p_returns = poisson[returns]
                # Accumulate rental probabilities, capping at init
                p_rentals_capped = np.zeros(n + 1)
                for rentals in range(n + 1):
                    actual = min(rentals, init)
                    p_rentals_capped[actual] += poisson[rentals]
                for actual_rentals in range(n + 1):
                    table[init, actual_rentals, returns] = p_rentals_capped[actual_rentals] * p_returns

        self.station_prob_tables[station_id] = table

    def build_transition_tables(self, s:StationState, a:int):
        """
        This method should return a tuple (X, Y) where
        X = p(s'| s, a) and is represented as a dictionary.
        The keys in the table will be s' (states) and the
        values will be p(s' | s, a).
        Y = E(r | s, a, s') should also be represented as a
        dictionary. The keys in the table will be s' (states)
        and the values will be E(r | s, a, s').

        For the purpose of this problem we will define the actions, a,
        as integers that can be positive or negative.  A negative number
        will mean we are moving a bikes from station 2 to station 1. A
        postive number will mean we are moving bikes from station 1 to station 2.
        You can ignore actions that result in more than 15 bikes at any station.
        You can also ignore actions that move more than 5 bikes from any one
        station to another.

        >>> s = YorkBikeRentalProblem(5,3,3)
        >>> a,b = s.build_transition_tables(StationState(2,3),2)
        >>> len(a)
        36
        >>> len(b)
        36
        >>> abs(float(a[StationState(0,5)]) - 0.029562047427443307) < 0.01
        True
        >>> abs(float(a[StationState(5,5)]) - 0.10969106850455339) < 0.01
        True
        >>> abs(float(a[StationState(3,2)]) - 0.017394719585080962) < 0.01
        True
        >>> abs(float(b[StationState(0,5)]) - 53.49) < 3
        True
        >>> abs(float(b[StationState(3,2)]) - 119.68) < 3
        True
        >>> abs(float(b[StationState(1,1)]) - 131.02) < 3
        True
        """
        n = self.max_bikes
        RENTAL_PRICE = 30  # $30 per rental (as per assignment spec)
        MOVE_COST = 6      # $6 per bike moved (matches updated doctests)

        # Apply action: positive a = move bikes from station1 to station2
        #               negative a = move bikes from station2 to station1
        bikes1_after_move = s.station1 - a
        bikes2_after_move = s.station2 + a

        # Clamp to valid range [0, max_bikes]
        bikes1_after_move = max(0, min(n, bikes1_after_move))
        bikes2_after_move = max(0, min(n, bikes2_after_move))

        # Cost of moving bikes
        move_cost = abs(a) * MOVE_COST

        trans = {}  # p(s'|s,a)
        rewards = {}  # E(r|s,a,s')

        prob_table_0 = self.station_prob_tables[0]
        prob_table_1 = self.station_prob_tables[1]

        for rentals1 in range(n + 1):
            for returns1 in range(n + 1):
                p1 = prob_table_0[bikes1_after_move, rentals1, returns1]
                if p1 == 0:
                    continue
                # actual rentals capped by available
                actual_rentals1 = min(rentals1, bikes1_after_move)
                reward1 = actual_rentals1 * RENTAL_PRICE
                bikes1_end = min(bikes1_after_move - actual_rentals1 + returns1, n)

                for rentals2 in range(n + 1):
                    for returns2 in range(n + 1):
                        p2 = prob_table_1[bikes2_after_move, rentals2, returns2]
                        if p2 == 0:
                            continue
                        actual_rentals2 = min(rentals2, bikes2_after_move)
                        reward2 = actual_rentals2 * RENTAL_PRICE
                        bikes2_end = min(bikes2_after_move - actual_rentals2 + returns2, n)

                        sp = StationState(bikes1_end, bikes2_end)
                        prob = p1 * p2
                        total_reward = reward1 + reward2 - move_cost

                        if sp not in trans:
                            trans[sp] = 0.0
                            rewards[sp] = 0.0

                        # E[r|s,a,s'] weighted accumulation
                        rewards[sp] += prob * total_reward
                        trans[sp] += prob

        # Normalize rewards: rewards[sp] currently = sum of prob*reward,
        # divide by trans[sp] to get E[r|s,a,s']
        for sp in trans:
            if trans[sp] > 0:
                rewards[sp] /= trans[sp]

        return trans, rewards

    def _get_valid_actions(self, s: StationState):
        """Return list of valid actions for a given state."""
        actions = []
        for a in range(-5, 6):
            b1 = s.station1 - a
            b2 = s.station2 + a
            if 0 <= b1 <= self.max_bikes and 0 <= b2 <= self.max_bikes:
                actions.append(a)
        return actions

    def policy_evaluation(self, V, pi, gamma, threshold):
        """
        >>> p = YorkBikeRentalProblem(5, 3, 3)
        >>> V = np.zeros((6, 6))
        >>> states = [StationState(s0, s1) for s0 in range(6) for s1 in range(6)]
        >>> pi = {s: 0 for s in states}
        >>> V = p.policy_evaluation(V, pi, gamma=0.9, threshold=0.5)
        >>> abs(round(float(V[0, 0]), 1) - 1258.5) < 15.0
        True
        >>> abs(round(float(V[5, 5]), 1) - 1474.2) < 15.0
        True
        >>> abs(round(float(V[0, 5]), 1) - 1365.9) < 15.0
        True
        >>> abs(round(float(V[5, 0]), 1) - 1366.5) < 15.0
        True
        """
        n = self.max_bikes
        states = [StationState(s0, s1) for s0 in range(n + 1) for s1 in range(n + 1)]

        while True:
            delta = 0.0
            for s in states:
                a = pi[s]
                trans, reward_exp = self.build_transition_tables(s, a)
                new_val = 0.0
                for sp, prob in trans.items():
                    r = reward_exp[sp]
                    new_val += prob * (r + gamma * V[sp.station1, sp.station2])
                old_val = V[s.station1, s.station2]
                V[s.station1, s.station2] = new_val
                delta = max(delta, abs(old_val - new_val))
            if delta < threshold:
                break
        return V

    def policy_improvement(self, V, pi, gamma):
        """
        >>> p = YorkBikeRentalProblem(5, 3, 3)
        >>> states = [StationState(s0, s1) for s0 in range(6) for s1 in range(6)]
        >>> pi = {s: 0 for s in states}
        >>> V = np.zeros((6, 6))
        >>> V = p.policy_evaluation(V, pi, gamma=0.9, threshold=0.5)
        >>> pi, stable = p.policy_improvement(V, pi, gamma=0.9)
        >>> pi[StationState(0, 0)]
        0
        >>> abs(pi[StationState(5, 0)] + pi[StationState(0, 5)]) < 0.00001
        True
        >>> pi[StationState(3, 3)]
        0
        """
        n = self.max_bikes
        states = [StationState(s0, s1) for s0 in range(n + 1) for s1 in range(n + 1)]
        stable = True

        for s in states:
            old_action = pi[s]
            valid_actions = self._get_valid_actions(s)
            best_action = old_action
            best_val = float('-inf')

            for a in valid_actions:
                trans, reward_exp = self.build_transition_tables(s, a)
                val = 0.0
                for sp, prob in trans.items():
                    r = reward_exp[sp]
                    val += prob * (r + gamma * V[sp.station1, sp.station2])
                if val > best_val:
                    best_val = val
                    best_action = a

            pi[s] = best_action
            if best_action != old_action:
                stable = False

        return pi, stable

    def policy_iteration(self, gamma, threshold):
        """
        >>> p = YorkBikeRentalProblem(5, 3, 3)
        >>> p.policy_iteration(0.9, 0.5)
        >>> p.curr_policy[StationState(0, 0)]
        0
        >>> abs(p.curr_policy[StationState(5, 0)] + p.curr_policy[StationState(0, 5)]) < 0.00001
        True
        >>> p.curr_policy[StationState(3, 3)]
        0
        """
        n = self.max_bikes
        states = [StationState(s0, s1) for s0 in range(n + 1) for s1 in range(n + 1)]
        pi = {s: 0 for s in states}
        V = np.zeros((n + 1, n + 1))

        while True:
            V = self.policy_evaluation(V, pi, gamma, threshold)
            pi, stable = self.policy_improvement(V, pi, gamma)
            if stable:
                break

        self.curr_policy = pi

    def value_iteration(self, gamma: float, threshold: float):
        """
        >>> v = YorkBikeRentalProblem(5, 3, 3)
        >>> v.value_iteration(0.9, 0.5)
        >>> v.curr_policy[StationState(0, 0)]
        0
        >>> abs(v.curr_policy[StationState(5, 0)] + v.curr_policy[StationState(0, 5)]) < 0.00001
        True
        >>> v.curr_policy[StationState(3, 3)]
        0
        """
        n = self.max_bikes
        states = [StationState(s0, s1) for s0 in range(n + 1) for s1 in range(n + 1)]
        V = np.zeros((n + 1, n + 1))

        while True:
            delta = 0.0
            for s in states:
                valid_actions = self._get_valid_actions(s)
                best_val = float('-inf')
                for a in valid_actions:
                    trans, reward_exp = self.build_transition_tables(s, a)
                    val = 0.0
                    for sp, prob in trans.items():
                        r = reward_exp[sp]
                        val += prob * (r + gamma * V[sp.station1, sp.station2])
                    if val > best_val:
                        best_val = val
                old_val = V[s.station1, s.station2]
                V[s.station1, s.station2] = best_val
                delta = max(delta, abs(old_val - best_val))
            if delta < threshold:
                break

        # Extract policy
        pi = {}
        for s in states:
            valid_actions = self._get_valid_actions(s)
            best_action = 0
            best_val = float('-inf')
            for a in valid_actions:
                trans, reward_exp = self.build_transition_tables(s, a)
                val = 0.0
                for sp, prob in trans.items():
                    r = reward_exp[sp]
                    val += prob * (r + gamma * V[sp.station1, sp.station2])
                if val > best_val:
                    best_val = val
                    best_action = a
            pi[s] = best_action

        self.curr_policy = pi

    def show_policy(self):
        for i in range(0, self.max_bikes + 1):
            for j in range(0, self.max_bikes + 1):
                print(f'({i}, {j}): {self.curr_policy[StationState(i,j)]}', end=" ")
            print()


if __name__ == '__main__':

    import doctest
    doctest.testmod()

    # Initialize environment
    max_bikes = 10
    
    york_bikes = YorkBikeRentalProblem(max_bikes, 3, 3)
    
    #policy iteration
    york_bikes.policy_iteration(0.9, 0.5)
    york_bikes.show_policy()
    
    #value iteration
    york_bikes.value_iteration(0.9, 0.5)
    york_bikes.show_policy()