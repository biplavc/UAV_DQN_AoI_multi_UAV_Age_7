# multi_UAV_Age_7

1. state space = [age at UAV, age at BS, update channel quality] as a 1D array.
2. actions     = two parts per action - (updated UAV, sampled users). Updated UAV will be based on age difference sum of users in that UAV and  sampled user will be with respect to each UAV

3. order of different simulation settings are -
    a) multi_UAV_age_1 -> select BS_capacity UAVs to update and choose UAV_capacity users only from those UAVs to sample.
    b) multi_UAV_age_2 -> select BS_capacity UAVs to update and choose UAV_capacity users from any 1 UAV to sample 

    Both 1 and 2 assume the resources of sampling, i.e. the sub-carriers for sampling are shared among UAVs so that only 1 UAV can sample at any slot. I wanted to remove them in the future but both multi_UAV_age_3 and multi_UAV_age_4 carried this property. Only in multi_UAV_age_5 was this removed.

    c) multi_UAV_age_3 -> select BS_capacity UAVs to update and UAV_capacity users from each UAV to sample. Sampling and updating step are fully independent. A bug remained where the sampling could only be done by one UAV at a time.
    d) multi_UAV_age_4 -> multi-UAV_age_3 had an error where UAVs or users having same age pr age difference were there, it would select the ones with the lower indexes which is not right. If UAVs/users have equal age/age difference, the selection should be random. Bug from multi_UAV_age_3 remained where the sampling could only be done by one UAV at a time.

    e) multi_UAV_age_5 -> at sampling step, each UAV can sample users independently in addition to the UAV updating.

    f) multi_UAV_age_6 -> at update step, all users under the same UAV have the same update loss. This makes DQN not perform that well compared to MAD for exp2 so going back to the previous arrangement where each user had unique packet sample and update loss, and will be called multi_UAV_age_7.

    g) multi_UAV_age_7 -> packet loss and sample loss unique. See multi_UAV_age_6.
    
    h) multi_UAV_age_6 -> at update step, all users under the same UAV have the same update loss. This makes DQN not perform that well compared to MAD for exp2 so going back to the previous arrangement where each user had unique packet sample and update loss.


