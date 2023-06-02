import numpy as np
import os
import random
import matplotlib.pyplot as plt

"""
state space
00.01|02.03.04
05.06|07.08.09
10.11.12.13.14
15|16.17|18.19
20|21.22|23.24

Representation of State = (Taxi_position, Passesnger in taxi or not?, Passsenger_position)
Total number of states = 650
"""

I = []  #used for plotting iterations
D = []  #used for plotting max-norm/policy loss
iterations = []
epsilon = 0.001
gamma = 0.9
Dest = 23
Walls = [(1,2), (6,7), (15,16), (17,18), (20,21), (22,23), (2,1), (7,6), (16,15), (18,17), (21,20), (23,22)]
Walls_ = [(2,3),(3,2),(12,13),(13,12),(22,23),(23,22),(32,33),(33,32),(7,8),(8,7),(17,18),(18,17),(27,28),(28,27),(37,38),(38,37),
          (25,26),(26,25),(35,36),(36,35),(45,46),(46,45),(55,56),(56,55),(60,61),(61,60),(70,71),(71,70),(80,81),(81,80),(90,91),(91,90),
          (63,64),(64,63),(73,74),(74,73),(83,84),(84,83),(93,94),(94,93),(67,68),(68,67),(77,78),(78,77),(87,88),(88,87),(97,98),(98,97)]
Term = []

def plotGraph(x, R, xL, yL, label, name):
    plt.plot(x, R)
    plt.xlabel(xL)
    plt.ylabel(yL)
    plt.title(label)
    plt.savefig(name + ".jpg")
    plt.close()

#------MDP State Space------#
States = []
Actions = ["East", "North", "South", "West", "Pickup", "Putdown"]
Transitions = {}
Rewards = {}
termination = ()
Start_States = []

def environment(n = 5, Passenger_loc=0, Destination=23, Walls = Walls):
    States = []
    Transitions = {}
    Rewards = {}
    for i in range(n*n):
        for j in range(n*n):
            if (i == j):
                States.append((i,True,j))
            States.append((i,False,j))
    for state in States:
        Transitions[state] = {}
        Rewards[state] = {}
        for k in Actions:
            Transitions[state][k] = {}
            Rewards[state][k] = {}
            St = []
            r = state[0]//n
            c = state[0]%n
            St.append((state[0],False,state[2]))
            if (state[0] == state[2]):
                St.append((state[0],True,state[2]))
            if (r-1 >= 0):
                St.append(((r-1)*n + c,False,state[2]))
                St.append(((r-1)*n + c,True,(r-1)*n + c))
            if (c-1 >= 0):
                St.append((r*n + c - 1,False,state[2]))
                St.append((r*n + c - 1,True,r*n + c - 1))
            if (r+1 <= n-1):
                St.append(((r+1)*n + c,False,state[2]))
                St.append(((r+1)*n + c,True,(r+1)*n + c))
            if (c+1 <= n-1):
                St.append((r*n + c + 1,False,state[2]))
                St.append((r*n + c + 1,True,r*n + c + 1))
            # print(St)
            for i in range(len(St)):
                S1 = state
                row1 = S1[0]//n
                col1 = S1[0]%n
                S2 = St[i]
                row2 = S2[0]//n
                col2 = S2[0]%n
                if (S1 == (Destination,False,Destination) and S2 != S1):
                    Transitions[state][k][S2] = 0
                    Rewards[state][k][S2] = 0
                    continue
                elif (S1 == (Destination,False,Destination) and S2 == S1):
                    Transitions[state][k][S2] = 1
                    Rewards[state][k][S2] = 0
                    continue
                #assigning rewards
                if (k == "Putdown" and S1[1] == True and S2[1] == False and S2[2] == S1[2] and S1[0] == S2[0] and S1[0] == Destination):
                    Rewards[state][k][S2] = 20
                elif (k == "Putdown" and (S1[0] != S2[0] or S1[0] != S1[2] or S2[0] != S2[2])):
                    Rewards[state][k][S2] = -10
                elif (k == "Pickup" and (S1[0] != S2[0] or S1[0] != S1[2] or S2[0] != S2[2])):
                    Rewards[state][k][S2] = -10
                else:
                    Rewards[state][k][S2] = -1
                #assigning transition probabilities
                if (S1 == S2 and k != "Pickup" and k != "Putdown"):
                    continue
                if (S1[0],S2[0]) in Walls:
                    Transitions[state][k][S2] = 0
                elif k == "Pickup":
                    if (S2[1] == True and S1[0] == S2[0] and S1[2] == S1[0] and S2[0] == S2[2] and S1[1] == False):
                        Transitions[state][k][S2] = 1
                    else:
                        Transitions[state][k][S2] = 0
                elif k == "Putdown":
                    if (S2[1] == False and S2[2] == S2[0] and S1[0] == S2[0] and S1[2] == S1[0] and S1[1] == True):
                        Transitions[state][k][S2] = 1
                    else:
                        Transitions[state][k][S2] = 0
                elif k == "North" and S1[1] == S2[1]:
                    if (S1[1] == False and S1[2] != S2[2]):
                        Transitions[state][k][S2] = 0
                    elif row1-1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.85
                    elif row1+1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1 == col2+1:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1+1 == col2:
                        Transitions[state][k][S2] = 0.05
                    else:
                        Transitions[state][k][S2] = 0
                elif k == "South" and S1[1] == S2[1]:
                    if (S1[1] == False and S1[2] != S2[2]):
                        Transitions[state][k][S2] = 0
                    elif row1+1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.85
                    elif row1-1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1 == col2+1:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1+1 == col2:
                        Transitions[state][k][S2] = 0.05
                    else:
                        Transitions[state][k][S2] = 0
                elif k == "East" and S1[1] == S2[1]:
                    if (S1[1] == False and S1[2] != S2[2]):
                        Transitions[state][k][S2] = 0
                    elif row1-1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1+1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1 == col2+1:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1+1 == col2:
                        Transitions[state][k][S2] = 0.85
                    else:
                        Transitions[state][k][S2] = 0
                elif k == "West" and S1[1] == S2[1]:
                    if (S1[1] == False and S1[2] != S2[2]):
                        Transitions[state][k][S2] = 0
                    elif row1-1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1+1==row2 and col1 == col2:
                        Transitions[state][k][S2] = 0.05
                    elif row1 == row2 and col1-1 == col2:
                        Transitions[state][k][S2] = 0.85
                    elif row1 == row2 and col1+1 == col2:
                        Transitions[state][k][S2] = 0.05
                    else:
                        Transitions[state][k][S2] = 0
                else:
                    Transitions[state][k][S2] = 0
            count = 0
            for st in St:
                if st != state:
                    count = count + Transitions[state][k][st]
            count = round(1-count, 2)
            if (count == 0.09):
                count = 0.1
            Transitions[state][k][state] = count
    termination = (Destination, False, Destination)
    return States, Actions, Transitions, Rewards, termination

def startStates(Destination = 23, depots={"R": 0, "G": 4, "Y": 20, "B": 23}):
    Start_States = []
    for depot1 in depots:
        for depot2 in depots:
            if depots[depot1] != Destination and depots[depot2] != Destination and depots[depot1] != depots[depot2]:
                Start_States.append((depots[depot1], False, depots[depot2]))
    return Start_States
#------MDP State Space------#



#------Simulator------#
def Simulator(state, k):
    step = random.random()
    start = 0
    for s in Transitions[state][k]:
        start += Transitions[state][k][s]
        if (step <= start):
            return Rewards[state][k][s],s
#------Simulator------#



#------Value Iteration------#
def ValueIteration(gamma):
    iteration = 0
    V0 = {}
    for state in States:
        V0[state] = 0
    while True:
        iteration += 1
        dist = 0
        V_next = V0.copy()
        for state in States:
            s = []
            for action in Actions:
                s.append(0)
                for state2 in Transitions[state][action]:
                    s[-1] += Transitions[state][action][state2]*(Rewards[state][action][state2] + gamma*V_next[state2])
            V0[state] = max(s)
            dist = max(dist, abs(V0[state] - V_next[state]))
        I.append(iteration)
        D.append(dist)
        if (dist < epsilon):
            print(iteration)
            return V_next

def get_policy(V, gamma):
    pi = {}
    for s in States:
        pi[s] = max(Actions, key=lambda a: sum([Transitions[s][a][s1] *(Rewards[s][a][s1]+ gamma*V[s1]) for s1 in Transitions[s][a]]))
    return pi

def policy_tester(V, si, gamma):
    # calculate total reward for value space V
    r_net = 0
    iteration = 1
    discount = 1
    while iteration <= 500:
        a = max(Actions, key=lambda a: sum([Transitions[si][a][s1] *(Rewards[si][a][s1] + gamma*V[s1]) for s1 in Transitions[si][a]]))
        r, sf = Simulator(si, a)
        r_net += discount*r
        if sf == termination:
            break
        si = sf
        iteration += 1
        discount *= gamma
    return r_net
#------Value Iteration------#



#------Policy Iteration------#
def policy_evaluation_alg(policy, gamma):
    X_mat = []
    Y_mat = []
    for state in States:
        x = []
        y = 0
        for state1 in States:
            if (state == state1):
                x.append(1-Transitions[state][policy[state]][state1]*gamma)
                y += Transitions[state][policy[state]][state1]*Rewards[state][policy[state]][state1]
            elif state1 in Transitions[state][policy[state]]:
                x.append(-Transitions[state][policy[state]][state1]*gamma)
                y += Transitions[state][policy[state]][state1]*Rewards[state][policy[state]][state1]
            else:
                x.append(0)
        X_mat.append(x)
        Y_mat.append(y)
    V_mat = np.linalg.solve(X_mat,Y_mat)
    V = {}
    for i in range(len(States)):
        V[States[i]] = V_mat[i]
    return V, V_mat

def policy_evaluation(policy, gamma):
    iteration = 0
    V0 = {}
    for state in States:
        V0[state] = 0
    while True:
        iteration += 1
        dist = 0
        V_next = V0.copy()
        for state in States:
            s = 0
            for state2 in Transitions[state][policy[state]]:
                s += Transitions[state][policy[state]][state2]*(Rewards[state][policy[state]][state2]+gamma*V_next[state2])
            V0[state] = s
            dist = max(dist, abs(V0[state] - V_next[state]))
        if (dist < epsilon):
            iterations.append(iteration)
            return V_next

def lookahead(V, state, gamma):
    Util = {}
    for action in Actions:
        Util[action] = 0
        for state2 in Transitions[state][action]:
            Util[action] += Transitions[state][action][state2] * (Rewards[state][action][state2] + gamma * V[state2])
    return Util

def exp_util(policy, V, gamma):
    U_pi = []
    for i in range(len(States)):
        u = 0
        for s2 in  Transitions[States[i]][policy[States[i]]]:
            u += Transitions[States[i]][policy[States[i]]][s2] * (Rewards[States[i]][policy[States[i]]][s2] + gamma * V[s2])
        U_pi.append(u)
    return U_pi

def policy_improvement(gamma, method = 1):
    policy = {}
    for state in States:
        policy[state] = "East"
    itern = 0
    pol = []
    Util = []
    V = []
    U = []
    while itern < 10:
        itern += 1
        V = policy_evaluation(policy, gamma) if method == 2 else policy_evaluation_alg(policy, gamma)[0]
        done  = True
        for state in States:
            act1 = policy[state]
            action_util = lookahead(V, state, gamma)
            act2 = max(action_util, key= lambda x: action_util[x])
            if act1 != act2:
                done = False
                policy[state] = act2
        pol.append(policy)
        Util.append(V)
        if done:
            break
    for s in States:
        U.append(V[s])
    for i in range(1,itern+1):
        I.append(i)
        U_pi = exp_util(pol[i-1], Util[i-1], gamma)
        diff = abs(np.array(U_pi)-np.array(U))
        D.append(max(diff))
    return policy, V
#------Policy Iteration------#



#------Q-learning------#
def QLearning(Q, e, alpha, gamma, episodes, freq, option):
    episode = 0
    R = [0 for i in range(int(episodes/freq) + 1)]
    si = Start_States[random.randint(0, len(Start_States) - 1)]
    for i in range(10):
        R[0] += policy_tester(Q, si, gamma)
    R[0] /= 10
    while episode < episodes:
        si = Start_States[random.randint(0, len(Start_States) - 1)]
        if option == 1:
            Q_SARS(Q, si, e, alpha, gamma, False)
        if option == 2:
            Q_SARS(Q, si, e, alpha, gamma, True)
        if option == 3:
            Q_SARSA(Q, si, e, alpha, gamma, False)
        if option == 4:
            Q_SARSA(Q, si, e, alpha, gamma, True)
        if episode % freq == freq - 1:
            si = Start_States[random.randint(0, len(Start_States) - 1)]
            for i in range(10):
                R[int((episode+1)/freq)] += policy_tester(Q, si, gamma)
            R[int((episode+1)/freq)] /= 10
        episode += 1
    return R

def Q_SARS(Q, si, e, alpha, gamma, decay):
    # implement one episode of Q with fixed e
    iteration = 1
    while iteration <= 500:
        e_rate = e/iteration if decay else e
        a = max(Actions, key=lambda a: Q[si][a]) if random.random() > e_rate else Actions[random.randint(0, len(Actions) - 1)]
        r, sf = Simulator(si, a)
        Q[si][a] = (1 - alpha)*Q[si][a] + alpha*(r + gamma*max([Q[sf][af] for af in Actions]))
        if sf == termination:
            break
        si = sf
        iteration += 1

def Q_SARSA(Q, si, e, alpha, gamma, decay):
    # implement one episode of Q with fixed e
    iteration = 1
    a = max(Actions, key=lambda a: Q[si][a]) if random.random() > e else Actions[random.randint(0, len(Actions) - 1)]
    while iteration <= 500:
        e_rate = e/iteration if decay else e
        r, sf = Simulator(si, a)
        af = max(Actions, key=lambda a: Q[si][a]) if random.random() > e_rate else Actions[random.randint(0, len(Actions) - 1)]
        Q[si][a] = (1 - alpha)*Q[si][a] + alpha*(r + gamma*(Q[sf][af]))
        if sf == termination:
            break
        si = sf
        a = af
        iteration += 1

def policy_tester(Q, si, gamma):
    # calculate total reward for a policy Q
    r_net = 0
    iteration = 1
    discount = 1
    while iteration <= 500:
        a = max(Actions, key=lambda a: Q[si][a])
        r, sf = Simulator(si, a)
        r_net += discount*r
        if sf == termination:
            break
        si = sf
        iteration += 1
        discount *= gamma
    return r_net
#------Q-learning------#



#------Part-B------#
def reward_vs_episodes(i, e, alpha, gamma, n, freq, plotName, fileName):
    Q = {state:{action:0 for action in Actions} for state in States}
    R = QLearning(Q, e, alpha, gamma, n, freq, i)
    E = [freq*i for i in range(int(n/freq) + 1)]
    plotGraph(E, R, "Number of Episodes", "Averaged Discount Value", plotName, fileName)
    return Q
#------Part-B------#



#------Main Function Calls------#
States, Actions, Transitions, Rewards, termination = environment()
Start_States = startStates()

'''
# Part A2(a)
print("Part A2(a): Value Iteration in 5X5 grid")
print("\nNumber of Iterations: ", end = "")
I = []
D = []
ValueIteration(0.9)
plotGraph(I, D, "Iteration", "Max-Norm Distance", "Value Iteration at Discount Factor, gamma = 0.9", "A1_Value_Iteration")
print("\n\n")
'''

'''
# Part A2(b)
print("Part A2(b): Effect of Discount Factor on Value Iteration Convergence")
discount = [0.01, 0.1, 0.5, 0.8, 0.99]
for gamma in discount:
    print("")
    print("Case#" + str(discount.index(gamma)+1))
    I = []
    D = []
    print("Number of Iterations: ", end = "")
    ValueIteration(gamma)
    plotGraph(I, D, "Iteration", "Max-Norm Distance", "Value Iteration vs Discount Factor, gamma = " + str(gamma), "A1_Discount_" + str(discount.index(gamma) + 1))
print("\n\n")
'''

'''
# Part A2(c)
print("Part A2(c): Effect of Initial Locations on Value Iteration")
States, Actions, Transitions, Rewards, termination = environment(5, 20, 4, Walls)
Start_States = startStates(4, {"R": 0, "G": 4, "Y": 20, "B": 23})
discount = [0.1, 0.99]
states_ = [(0, False, 20), (20, False, 23)]
for si in states_:
    for gamma in discount:
        print("")
        print("Case#" + str(discount.index(gamma)+2*states_.index(si)+1))
        state = si
        print("Number of Iterations: ", end = "")
        V = ValueIteration(gamma)
        pi = get_policy(V, gamma)
        print(str(state) + ": " + str(pi[state]))
        for r in range(20):
            _,state = Simulator(state, pi[state])
            print(str(state) + ": " + str(pi[state]))
print("\n\n")
'''

'''
States, Actions, Transitions, Rewards, termination = environment()
Start_States = startStates()
'''

'''
# Part A3(a)
print("Part A3(a): Algebraic and Iterative Methods for Policy Evaluation")
print("")
print("Case#1")
I = []
D = []
policy_improvement(0.9, 1)
print("Algebraic Policy Evaluation")
print("Total Policy Iterations = " + str(len(I)))
print("")
print("Case#2")
iterations = []
I = []
D = []
policy_improvement(0.9, 2)
print("Iterative Policy Evaluation")
print("Total Policy Iterations = " + str(len(I)))
print("Total Policy Evaluations = " + str(sum(iterations)))
print("Policy Evaluations per Iteration = " + str(sum(iterations)/len(I)))
print("Policy Evaluations vs Iteration", iterations)
print("\n\n")
'''

'''
# Part A3(b)
print("Part A3(b): Effect of Discount Factor on Policy Iteration Convergence")
discount = [0.01, 0.1, 0.5, 0.8, 0.99]
for gamma in discount:
    print("")
    print("Case#" + str(discount.index(gamma)+1))
    print("Discount Factor = " + str(gamma))
    I = []
    D = []
    policy_improvement(gamma)
    plotGraph(I, D, "Iteration", "Policy Loss", "Policy Iteration vs Discount Factor, gamma = " + str(gamma), "A3_Discount_" + str(discount.index(gamma) + 1))
print("\n\n")
'''

'''
# Part B2
print("Part B2: Convergence of Q-Learning Algorithms")
for i in range(1, 5):
    print("")
    print("Case#" + str(i))
    labels = ["Q-Learning", "Q-Learning with decay", "SARSA", "SARSA with decay"]
    print(labels[i-1])
    reward_vs_episodes(i, 0.1, 0.25, 0.99, 2000, 10, "Algorithm Convergence for " + labels[i-1], "B2_" + str(i))
print("\n\n")
'''


# Part B3
i = 1
Q = {state:{action:0 for action in Actions} for state in States}
R = QLearning(Q, 0.1, 0.25, 0.99, 2000, 10, i)
print("Part B3: Effect of depot locations on reward values using Q-Learning")
s = []
for i in range(5):
    print("")
    print("Case#" + str(i+1))
    si = Start_States[random.randint(0, len(Start_States) - 1)]
    while si in s:
        si = Start_States[random.randint(0, len(Start_States) - 1)]
    print("Taxi Start Location: (" + str(si[0]%5) + "," + str(4 - si[0]//5) + ")")
    print("Passenger Pickup Location: (" + str(si[2]%5) + "," + str(4 - si[2]//5) + ")")
    r = 0
    for j in range(10):
        r += policy_tester(Q, si, 0.99)
    r /= 10
    print("Averaged Rewards Earned = " + str(r))
print("\n\n")


'''
# Part B4
i = 1
print("Part B4: Effect of Exploration and Learning Rates on Q-Learning Algorithm Convergence")
exploration = [0, 0.05, 0.1, 0.5, 0.9]
for j in range(len(exploration)):
    print("")
    print("Case#" + str(j+1))
    print("Exploration Rate = " + str(exploration[j]) + ", Learning Rate = 0.1")
    reward_vs_episodes(i, exploration[j], 0.1, 0.99, 2000, 10, "Q-Learning vs Exploration Rate, e = " + str(exploration[j]), "B4_Exploration_" + str(j + 1))
learning = [0.1, 0.2, 0.3, 0.4, 0.5]
for j in range(len(learning)):
    print("")
    print("Case#" + str(j+6))
    print("Exploration Rate = 0.1, Learning Rate = " + str(learning[j]))
    reward_vs_episodes(i, 0.1, learning[j], 0.99, 2000, 10, "Q-Learning vs Learning Rate, a = " + str(learning[j]), "B4_Learning_" + str(j + 1))
print("\n\n")
'''

'''
# Part B5
i = 1
depots = {'R':0, 'G':5, 'C':8, 'W':33, 'M':46, 'Y':80, 'B':94, 'P':99}
Destinations = [0, 0, 0, 99, 99, 99]
Starts = [(80,False,33), (80,False,94), (46,False,33), (80,False,33), (80,False,94), (46,False,33)]
print("Part B5: Effect of source and destination on aggregate reward values in 10X10 grid")
f = 80
for i in range(6):
    print("")
    print("Case#" + str(i+1))
    si = Starts[i]
    print("Taxi Start Location: (" + str(si[0]%10) + "," + str(9 - si[0]//10) + ")")
    print("Passenger Pickup Location: (" + str(si[2]%10) + "," + str(9 - si[2]//10) + ")")
    print("Passenger Destination: (" + str(Destinations[i]%10) + "," + str(9 - Destinations[i]//10) + ")")
    States, Actions, Transitions, Rewards, termination = environment(10,si[2],Destinations[i],Walls_)
    Start_States = startStates(Destinations[i],depots)
    Q = {state:{action:0 for action in Actions} for state in States}
    R = QLearning(Q, 0.1, 0.25, 0.99, 10000, f, i)
    if i == 2:
        E = [f*i for i in range(int(10000/f) + 1)]
        plotGraph(E, R, "Number of Episodes", "Averaged Discount Value", "Q-Learning Convergence for 10X10 grid", "B5_Convergence")
    r = 0
    for j in range(10):
        r += policy_tester(Q, si, 0.99)
    r /= 10
    print("Averaged Rewards Earned = " + str(r))
'''
#------Main Function Calls------#
