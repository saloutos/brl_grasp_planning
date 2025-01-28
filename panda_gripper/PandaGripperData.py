# imports
import numpy as np

# joint data class
class JointData:
    def __init__(self, name=""):
        self.name = name
        # NOTE: all joint data should be stored as 1-D numpy arrays
        # TODO: should remove some of these
        self.q = np.array([0.0])
        self.qd = np.array([0.0])
        self.tau = np.array([0.0])
        self.ctrl = np.array([0.0])
        # TODO: set default values for controllers to use?
    # joint control law
    def update_control(self):
        # TODO: implement something here?
        return self.ctrl # not sure if the return is necessary
    # logging functions
    def log_data(self):
        # TODO: include gains?
        return [self.q[0], self.qd[0], self.tau[0], self.ctrl[0]]
    def log_header(self):
        # TODO: include gains?
        return [self.name+"_q",self.name+"_qd",self.name+"_tau",
                self.name+"_ctrl"]

# sensor data classes
class SensorData:
    def __init__(self, name=""):
        self.name = name
        self.kinematics = (np.zeros((3,)), np.eye(3)) # (pos, R) of sensor frame, in world frame
    # logging functions
    def log_data(self):
        return []
    def log_header(self):
        return []
    # processing sensor data
    def update_kinematics(self, pos, R): # mostly for visualization
        self.kinematics = (pos, R)
    def update_raw_data_from_hw(self):
        pass
    def update_raw_data_from_sim(self):
        pass
    def process_data(self):
        pass
    def sync_data_to_viewer(self, scene, start_idx):
        return start_idx


# gripper data class
class PandaGripperData:
    # TODO: initialize with lists of joints and sensors? rather than hardcoding?
    # TODO: if init takes joints and sensors, should it take indices too? those could be added in platform?
    def __init__(self): #, joints=[], sensors=[]):
        # list of joints
        # just one actuator, just two joints that are coupled?
        # TODO: might need to do something about this
        joints = [JointData("finger_joint1")]
        # list of sensors
        sensors = []
        # NOTE: these should match the names of the joints and sensors in the mujoco model
        # now, store in dicts so that we have named access
        self.joint_names = [joint.name for joint in joints]
        self.joints = dict(zip(self.joint_names, joints))
        # TODO: uncomment this when we have sensors
        self.sensor_names = [] #[sensor.name for sensor in sensors]
        self.sensors = {} #dict(zip(self.sensor_names, sensors))
        # other joint vars
        self.nj = len(joints) # number of joints
        self.ns = len(sensors) # number of sensors
        self.all_idxs = list(range(self.nj)) # all joint idxs

        # TODO: store kinematics in dict as well?
        # TODO: make "body kinematics data" class that can return T, R, pos, quat, etc?
        # TODO: then, can define list of bodies whose kinematics we want to track, and update all at once in GP
        # TODO: how likely is it we will want to track sensor kinematics for things beyond visualization?
        self.kinematics = {}
        self.kinematics['base'] = {'p': np.zeros((3,)), 'R':np.eye(3)} # pos, R, in world frame
        self.kinematics['left_finger'] = {'p':np.zeros((3,)), 'R':np.eye(3)} # pos, R, in world frame
        self.kinematics['right_finger'] =  {'p':np.zeros((3,)), 'R':np.eye(3)}

        # TODO: this doesn't feel like the right way to do this
        self.kinematics['base_des'] = {'p': np.zeros((3,)), 'R':np.eye(3)} # desired base pos, R in world frame (for mocap body)

        # TODO: other useful vars? don't need to populate them here, but could store them here
        # contact points?
        # have a separate "FingerData" class that has all the data for a finger?
        # these could be stored in controllers too
        # these should all be available from mujoco data, so maybe not necessary here

    # logging functions
    def log_data(self):
        line = []
        # TODO: neeed to sort keys of the dicts to ensure consistent order?
        for key in self.joints.keys():
            line += self.joints[key].log_data()
        # TODO: uncomment this when we have sensors
        # for key in self.sensors.keys():
        #     line += self.sensors[key].log_data()
        return line
    def log_header(self):
        header = []
        for key in self.joints.keys():
            header += self.joints[key].log_header()
        # TODO: uncomment this when we have sensors
        # for key in self.sensors.keys():
        #     header += self.sensors[key].log_header()
        return header

    # NOTE: getters and setters need to be in this class to that GD can be passed to controllers
    # NOTE: syncing GD and mjData will happen in the gripper platform class based on mode
    # getting and setting joint data by index (general versions)
    def get_joint_data(self, var_name, joint_idxs):
        # var name is string, joint_idxs is list of ints
        # iterate over joint_idxs, getting key from indexing list of joint names
        data = [self.joints[self.joint_names[idx]].__dict__[var_name][0] for idx in joint_idxs]
        return np.array(data)
    def set_joint_data(self, var_name, joint_idxs, data):
        # var name is string, joint_idxs is list of ints, data is 1-D array
        if len(joint_idxs)!=len(data):
            raise ValueError("Joint indices and data must be the same length.")
        else:
            # iterate over data, getting key from indexing list of joint names by indexing joint indexes
            for i, d in enumerate(data):
                self.joints[self.joint_names[joint_idxs[i]]].__dict__[var_name][0] = d

    # getting and setting specific joint data by index
    # TODO: add or remove these as necessary
    def get_q(self, joint_idxs):
        return self.get_joint_data("q", joint_idxs)
    def set_q(self, joint_idxs, data):
        self.set_joint_data("q", joint_idxs, data)
    def get_qd(self, joint_idxs):
        return self.get_joint_data("qd", joint_idxs)
    def set_qd(self, joint_idxs, data):
        self.set_joint_data("qd", joint_idxs, data)
    def get_tau(self, joint_idxs):
        return self.get_joint_data("tau", joint_idxs)
    def set_tau(self, joint_idxs, data):
        self.set_joint_data("tau", joint_idxs, data)
    def get_ctrl(self, joint_idxs):
        return self.get_joint_data("ctrl", joint_idxs)
    def set_ctrl(self, joint_idxs, data):
        self.set_joint_data("ctrl", joint_idxs, data)

    # updating joint control laws
    def update_all_joint_control(self):
        for key in self.joints.keys():
            self.joints[key].update_control()

    # # updating sensor data
    # def update_all_sensor_kinematics(self, all_data={}):
    #     for key in self.sensors.keys():
    #         self.sensors[key].update_kinematics(all_data[key][0], all_data[key][1])
    # def update_all_raw_sensor_data_from_hw(self, all_data={}):
    #     for key in self.sensors.keys():
    #         self.sensors[key].update_raw_data_from_hw(all_data[key])
    # def update_all_raw_sensor_data_from_sim(self, all_data={}):
    #     for key in self.sensors.keys():
    #         self.sensors[key].update_raw_data_from_sim(all_data[key])
    # def process_all_sensor_data(self):
    #     for key in self.sensors.keys():
    #         self.sensors[key].process_data()