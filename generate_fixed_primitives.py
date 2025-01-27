import numpy as np
import yaml

#  goal output:

# box_1: # also serves as object name
#   type: 0 # 0 is box, 1 is cylinder, 2 is sphere
#   size: [0.1, 0.1, 0.1]
#   mass: 0.1
#   rgba: [0.9, 0.1, 0.1, 1.0]
#   friction: [1.0, 0.02, 0.0005]
#   pos: [0, 0, 0.2]
#   # quat: [0, 0, 0, 1]
#   rpy: [0, 0, 45] # in degrees

# common densities:
# wood: 320-850 kg/m3
# water: 1000 kg/m3
# steel: 7800 kg/m3
# aluminum: 2700 kg/m3

# cereal box: 100 kg/m3
# other box: 240 kg/m3
# "cup" effective density: 200 kg/m3

# box sizes are half-lengths
# cylinder size is radius and half-height
# sphere size is radius

# object distribution:
fixed_types_sizes = [
# type, size, densities (kg/m3)
(0, [0.03, 0.03, 0.03],     [250, 500]), # medium cube
(0, [0.04, 0.04, 0.04],     [250, 500]), # large cube
(0, [0.02, 0.02, 0.04],     [250, 500]), # small 1x1x2 box
(0, [0.03, 0.03, 0.06],     [250, 500]), # medium 1x1x2 box
(0, [0.05, 0.05, 0.10],     [100, 250]), # large 1x1x2 box
(0, [0.02, 0.04, 0.06],     [100, 250]), # small 1x2x3 box
(0, [0.035, 0.07, 0.105],   [100, 250]), # medium 1x2x3 box
(0, [0.04, 0.10, 0.125],    [100, 250]), # large 1x2x3 box
(0, [0.035, 0.035, 0.105],  [100, 250]), # medium 1x1x3 box
(0, [0.045, 0.045, 0.125],  [100, 250]), # large 1x1x3 box

(1, [0.025, 0.05, 0],       [250, 500]), # small short cylinder
(1, [0.025, 0.075, 0],      [250, 500]), # small medium cylinder
(1, [0.025, 0.1, 0],        [100, 250]), # small tall cylinder
(1, [0.035, 0.075, 0],      [100, 250]), # medium medium cylinder
(1, [0.04, 0.08, 0],        [100, 250]), # medium medium cylinder #2
(1, [0.035, 0.1, 0],        [100, 250]), # medium tall cylinder
(1, [0.035, 0.11, 0],       [250, 500]), # medium tall cylinder #2
(1, [0.04, 0.12, 0],        [100, 250]), # medium tall cylinder #3
(1, [0.05, 0.05, 0],        [100, 250]), # large short cylinder
(1, [0.05, 0.075, 0],        [100, 250]), # large medium cylinder

(2, [0.025, 0, 0],          [250, 500]), # small sphere
(2, [0.03, 0, 0],           [250, 500]), # small sphere
(2, [0.04, 0, 0],           [100, 250]), # medium sphere
(2, [0.05, 0, 0],           [250, 500]), # medium sphere
]

num_boxes = 0
num_cylinders = 0
num_spheres = 0

# iterate and save
for i in range(len(fixed_types_sizes)):
    type_choice = fixed_types_sizes[i][0]
    size_choice = fixed_types_sizes[i][1]
    densities = fixed_types_sizes[i][2]
    for j in range(len(densities)):
        # set default values for pos and orientation
        temp = {'pos': [0, 0, 0], 'rpy': [0, 0, 0], 'friction': [1.0, 0.02, 0.0005]}
        # choose type, mass
        temp['type'] = type_choice

        # based on type, choose size and set name
        if type_choice == 0:
            num_boxes += 1
            temp['size'] = fixed_types_sizes[i][1]
            volume = (2*size_choice[0]) * (2*size_choice[1]) * (2*size_choice[2])
            temp['mass'] = densities[j] * volume
            out_name = 'box_' + str(num_boxes)
        elif type_choice == 1:
            num_cylinders += 1
            temp['size'] = fixed_types_sizes[i][1]
            volume = np.pi * size_choice[0] * size_choice[0] * (2*size_choice[1])
            temp['mass'] = densities[j] * volume
            out_name = 'cylinder_' + str(num_cylinders)
        else:
            num_spheres += 1
            temp['size'] = fixed_types_sizes[i][1]
            volume = (4/3) * np.pi * size_choice[0] * size_choice[0] * size_choice[0]
            temp['mass'] = densities[j] * volume
            out_name = 'sphere_' + str(num_spheres)
        # choose rgb vals
        temp['rgba'] = [float(np.random.rand(1)),
                        float(np.random.rand(1)),
                        float(np.random.rand(1))] + [1.0]
        # write to yaml
        out = {out_name: temp}
        with open('primitives/single_objects/fixed/'+out_name+'.yaml', 'a') as file:
            yaml.dump(out, file)

        # test:
        with open('primitives/single_objects/fixed/'+out_name+'.yaml', 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            print(data)
