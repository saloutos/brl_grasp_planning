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

types = np.array([0, 1, 2])
sizes = np.array([0.01, 0.02, 0.03, 0.05, 0.07])
masses = np.array([0.025, 0.05, 0.075, 0.1, 0.3, 0.5])

num_choices = 50

num_boxes = 0
num_cylinders = 0
num_spheres = 0

for i in range(num_choices):
    # set default values for pos and orientation
    temp = {'pos': [0, 0, 0], 'rpy': [0, 0, 0], 'friction': [1.0, 0.02, 0.0005]}
    # choose type, mass
    type_choice = np.random.choice(types)
    mass_choice = np.random.choice(masses)
    temp['type'] = int(type_choice)
    temp['mass'] = float(mass_choice)
    # based on type, choose size and set name
    if type_choice == 0:
        num_boxes += 1
        temp['size'] = [float(np.random.choice(sizes)),
                        float(np.random.choice(sizes)),
                        float(np.random.choice(sizes))]
        out_name = 'box_' + str(num_boxes)
    elif type_choice == 1:
        num_cylinders += 1
        temp['size'] = [float(np.random.choice(sizes)),
                        float(np.random.choice(sizes)),
                        0]
        out_name = 'cylinder_' + str(num_cylinders)
    else:
        num_spheres += 1
        temp['size'] = [float(np.random.choice(sizes)), 0, 0]
        out_name = 'sphere_' + str(num_spheres)
    # choose rgb vals
    temp['rgba'] = [float(np.random.rand(1)),
                    float(np.random.rand(1)),
                    float(np.random.rand(1))] + [1.0]
    # write to yaml
    out = {out_name: temp}
    with open('primitives/single_objects/random/'+out_name+'.yaml', 'a') as file:
        yaml.dump(out, file)

    # test:
    with open('primitives/single_objects/random/'+out_name+'.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        print(data)
