from collections import Counter

def get_reward(selection,state):

    reward = 0
    num_mix_slots = len(state)-2
    Mix_count = Counter(state[0:num_mix_slots])
    Hidden_mix = 16

#######################################################
    #Penalizations

    for mix in Mix_count:
        if Mix_count[mix] > 1:
            if mix != Hidden_mix:
                reward -= 20 #Penalize for showing the same mix more than once

    reward += Mix_count[Hidden_mix]*2 # Reward for hiding mix

    if Mix_count[Hidden_mix] == num_mix_slots:
        return -60 #Penalize for hiding all mixes

######################################################

    #Mixture
    for i, mix in enumerate(state[0:num_mix_slots]):
        if mix == selection[0]:
            reward += 20 - i*2 #Reward for showing desired mix
            break
        reward -= 30    #Penalize for not showing the desired mix

#######################################################

    #Additive
    if state[-2] == selection[1]:    
        reward +=10 #Reward for guessing the right additive
    else:
        reward += - abs(state[-2]-selection[1]) * 2 #Penalize for guessing wrong additive

#######################################################

    #Container
    if state[-1] == selection[2]: 
        reward += 10    #Reward for guessing the right container
    else: reward += -10 #Penalize for guessing wrong container

#######################################################

    return reward 
