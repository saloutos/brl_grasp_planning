# imports
import numpy as np
import os
import time
import copy
import dill
import seaborn as sns
import matplotlib.pyplot as plt

# copy file here
summary = [
            ['cylinder_3', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.5333333333333333, 0.45, 0.4564102564102564], 113.67806911468506],
            ['cylinder_3', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [1.0, 0.9944444444444445, 0.9948717948717949], 101.75257110595703],
            ['cylinder_3', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [1.0, 0.8305555555555556, 0.8435897435897436], 117.79092359542847],
            ['cylinder_3', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [0.9333333333333333, 0.825, 0.8333333333333334], 99.90404415130615],
            ['cylinder_3B', ([0, 0, 0.08], [90, 0, 0]), 'CGN', 30, [0.5, 0.4111111111111111, 0.41794871794871796], 124.68107461929321],
            ['cylinder_3B', ([0, 0, 0.08], [90, 0, 0]), 'EDGE', 30, [1.0, 0.9611111111111111, 0.9641025641025641], 107.95523118972778],
            ['cylinder_3B', ([0, 0, 0.08], [90, 0, 0]), 'GSN', 30, [0.9666666666666667, 0.8138888888888889, 0.8256410256410256], 124.36203646659851],
            ['cylinder_3B', ([0, 0, 0.08], [90, 0, 0]), 'GIGA', 30, [0.7, 0.6277777777777778, 0.6333333333333333], 110.02759265899658],
            ['cylinder_6', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.5333333333333333, 0.4666666666666667, 0.4717948717948718], 113.91707992553711],
            ['cylinder_6', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [1.0, 0.9916666666666667, 0.9923076923076923], 102.9404559135437],
            ['cylinder_6', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [1.0, 0.8416666666666667, 0.8538461538461538], 118.01836585998535],
            ['cylinder_6', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [1.0, 0.9972222222222222, 0.9974358974358974], 100.91162061691284],
            ['cylinder_6B', ([0, 0, 0.08], [90, 0, 0]), 'CGN', 30, [0.7333333333333333, 0.6527777777777778, 0.658974358974359], 118.23467445373535],
            ['cylinder_6B', ([0, 0, 0.08], [90, 0, 0]), 'EDGE', 30, [1.0, 0.9666666666666667, 0.9692307692307692], 106.21994352340698],
            ['cylinder_6B', ([0, 0, 0.08], [90, 0, 0]), 'GSN', 30, [1.0, 0.8638888888888889, 0.8743589743589744], 122.29190444946289],
            ['cylinder_6B', ([0, 0, 0.08], [90, 0, 0]), 'GIGA', 30, [0.7333333333333333, 0.6805555555555556, 0.6846153846153846], 108.97609353065491],
            ['box_7', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.7333333333333333, 0.5166666666666667, 0.5333333333333333], 111.16702580451965],
            ['box_7', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [0.9666666666666667, 0.8638888888888889, 0.8717948717948718], 104.73284554481506],
            ['box_7', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [0.9, 0.775, 0.7846153846153846], 117.5351231098175],
            ['box_7', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [0.8333333333333334, 0.7111111111111111, 0.7205128205128205], 100.36771202087402],
            ['box_7B', ([0, 0, 0.08], [90, 0, 0]), 'CGN', 30, [0.6, 0.4888888888888889, 0.49743589743589745], 115.40882205963135],
            ['box_7B', ([0, 0, 0.08], [90, 0, 0]), 'EDGE', 30, [0.9666666666666667, 0.9222222222222223, 0.9256410256410257], 106.27959442138672],
            ['box_7B', ([0, 0, 0.08], [90, 0, 0]), 'GSN', 30, [1.0, 0.9694444444444444, 0.9717948717948718], 116.5389232635498],
            ['box_7B', ([0, 0, 0.08], [90, 0, 0]), 'GIGA', 30, [0.36666666666666664, 0.3472222222222222, 0.3487179487179487], 108.30581641197205],
            ['box_9', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.1, 0.09722222222222222, 0.09743589743589744], 112.55509686470032],
            ['box_9', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [1.0, 0.8027777777777778, 0.8179487179487179], 111.7677595615387],
            ['box_9', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [0.8333333333333334, 0.5861111111111111, 0.6051282051282051], 114.4700140953064],
            ['box_9', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [0.5333333333333333, 0.5305555555555556, 0.5307692307692308], 101.27432680130005],
            ['box_9B', ([0, 0, 0.08], [90, 0, 0]), 'CGN', 30, [0.1, 0.14722222222222223, 0.14358974358974358], 114.32179117202759],
            ['box_9B', ([0, 0, 0.08], [90, 0, 0]), 'EDGE', 30, [0.9666666666666667, 0.8583333333333333, 0.8666666666666667], 108.88619780540466],
            ['box_9B', ([0, 0, 0.08], [90, 0, 0]), 'GSN', 30, [0.6, 0.525, 0.5307692307692308], 117.72292423248291],
            ['box_9B', ([0, 0, 0.08], [90, 0, 0]), 'GIGA', 30, [0.23333333333333334, 0.2833333333333333, 0.2794871794871795], 105.73064923286438],
            ['sphere_4', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.7333333333333333, 0.5388888888888889, 0.5538461538461539], 98.9334180355072],
            ['sphere_4', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [1.0, 0.9111111111111111, 0.9179487179487179], 91.67236614227295],
            ['sphere_4', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [1.0, 0.7416666666666667, 0.7615384615384615], 104.63355469703674],
            ['sphere_4', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [0.9666666666666667, 0.7805555555555556, 0.7948717948717948], 90.9965033531189],
            ['sphere_5', ([0, 0, 0.08], [0, 0, 0]), 'CGN', 30, [0.4666666666666667, 0.4027777777777778, 0.4076923076923077], 101.73129940032959],
            ['sphere_5', ([0, 0, 0.08], [0, 0, 0]), 'EDGE', 30, [0.8333333333333334, 0.5611111111111111, 0.5820512820512821], 95.55154061317444],
            ['sphere_5', ([0, 0, 0.08], [0, 0, 0]), 'GSN', 30, [1.0, 0.6666666666666666, 0.6923076923076923], 105.54926538467407],
            ['sphere_5', ([0, 0, 0.08], [0, 0, 0]), 'GIGA', 30, [1.0, 0.8444444444444444, 0.8564102564102564], 87.21189093589783],
]

objects = []
planners = []
for i, data in enumerate(summary):
    objects.append(data[0])
    planners.append(data[2])

objects = list(set(objects))
planners = list(set(planners))
objects.sort()
planners.sort()
num_objects = len(objects)
num_planners = len(planners)

print(num_objects, objects)
print(num_planners, planners)



base_success_rates = np.zeros((num_objects, num_planners)) # 30 trials
pert_success_rates = np.zeros((num_objects, num_planners)) # 360 trials
all_success_rates = np.zeros((num_objects, num_planners)) # 390 trials

for i, data in enumerate(summary):
    obj_name = data[0]
    planner_name = data[2]
    success_rates = data[4]
    obj_index = objects.index(obj_name)
    planner_index = planners.index(planner_name)
    base_success_rates[obj_index, planner_index] = success_rates[0]
    pert_success_rates[obj_index, planner_index] = success_rates[1]
    all_success_rates[obj_index, planner_index] = success_rates[2]

base_mean_object = np.expand_dims(np.mean(base_success_rates, axis=1), 0).T
base_mean_planner = np.mean(base_success_rates, axis=0)

pert_mean_object = np.expand_dims(np.mean(pert_success_rates, axis=1), 0).T
pert_mean_planner = np.mean(pert_success_rates, axis=0)

all_mean_object = np.expand_dims(np.mean(all_success_rates, axis=1), 0).T
all_mean_planner = np.mean(all_success_rates, axis=0)





# print("Base Success Rates")
# print(base_success_rates)
print("mean base success rate per object:")
print(base_mean_object.T)
print("mean base success rate per planner:")
print(base_mean_planner)
print("")

# print("Perturbation Success Rates")
# print(pert_success_rates)
print("mean pert success rate per object:")
print(pert_mean_object.T)
print("mean pert success rate per planner:")
print(pert_mean_planner)
print("")

# print("All Success Rates")
# print(all_success_rates)
# print("mean all success rate per object:")
# print(all_mean_object)
# print("mean all success rate per planner:")
# print(all_mean_planner)
# print("")

# delta pert

delta_pert = pert_success_rates - base_success_rates
delta_pert_mean_object = pert_mean_object-base_mean_object
delta_pert_mean_planner = pert_mean_planner-base_mean_planner

# print("Delta Perturbation Success Rates")
# print(delta_pert)
print("delta mean pert success rate per object:")
print(delta_pert_mean_object.T)
print("delta mean pert success rate per planner:")
print(delta_pert_mean_planner)
print("")

# print("Relative change in success rates under perturbation")
rel_delta_pert = delta_pert / base_success_rates
rel_delta_pert_mean_object = delta_pert_mean_object / base_mean_object
rel_delta_pert_mean_planner = delta_pert_mean_planner / base_mean_planner

# print("Relative Delta Perturbation Success Rates")
# print(rel_delta_pert)
print("Relative delta mean pert success rate per object:")
print(rel_delta_pert_mean_object.T)
print("Relative delta mean pert success rate per planner:")
print(rel_delta_pert_mean_planner)
print("")

plt.figure()
sns.heatmap(base_success_rates, vmin=0.0, vmax=1.0, cmap='rocket', xticklabels=planners, yticklabels=objects, annot=True, fmt=".2f")
plt.title("Base Success Rates")

plt.figure()
sns.heatmap(pert_success_rates, vmin=0.0, vmax=1.0, cmap='rocket', xticklabels=planners, yticklabels=objects, annot=True, fmt=".2f")
plt.title("Perturbation Success Rates")

# plt.figure()
# sns.heatmap(all_success_rates, vmin=0.0, vmax=1.0, cmap='rocket', xticklabels=planners, yticklabels=objects, annot=True, fmt=".2f")
# plt.title("All Success Rates")

plt.figure()
sns.heatmap(delta_pert, cmap='crest', xticklabels=planners, yticklabels=objects, annot=True, fmt=".2f")
plt.title("Delta Perturbation Success Rates")

plt.figure()
sns.heatmap(rel_delta_pert, cmap='crest', xticklabels=planners, yticklabels=objects, annot=True, fmt=".2f")
plt.title("Relative Delta Perturbation Success Rates")

plt.show()