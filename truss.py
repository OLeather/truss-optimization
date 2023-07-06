from enum import Enum
import math
import matplotlib.pyplot as plt

# class syntax
class JointType(Enum):
    PIN = 1
    ROLLER = 2

# functional syntax
JointType = Enum('JointType', ['PIN', 'ROLLER'])

class Joint():
    def __init__(self, type, x, y, fx = 0, fy = 0):
        self.type = type
        self.x = x
        self.y = y
        self.update_forces(fx, fy)
    
    def update_forces(self, fx, fy):
        self.fx = fx
        self.fy = fy

class Force():
    def __init__(self, magnitude, direction, joints_applied):
        self.x = magnitude*math.cos(direction)
        self.y = magnitude*math.sin(direction)
        self.joints_applied = joints_applied
    
    def __init__(self, fx, fy, joints_applied):
        self.fx = fx
        self.fy = fy
        self.joints_applied = joints_applied

class Truss():
    MAX_TENSILE = 9 # kN
    MAX_COMPRESSIVE = 6 # kN

    def __init__(self, joints, links, forces):
        '''
        @param joints: list of Joint objects
        @param links: list of tuples connecting joints by index
        '''
        self.joints = joints
        self.links = links
        self.forces = forces
    
    def solve(self):
        '''
        @return : whether the truss survives the load
        '''
        return False

    def apply_external_forces(self):
        for force in self.forces:
            if len(force.joints_applied) == 1:
                self.joints[force.joints_applied[0]].update_forces(force.fx, force.fy)
            else:    
                total_len = abs(self.joints[force.joints_applied[-1]].x - self.joints[force.joints_applied[0]].x)
                force_per_len = float(force.fy) / total_len
                for i in range(len(force.joints_applied)-1):
                    joint0 = self.joints[force.joints_applied[i]]
                    joint1 = self.joints[force.joints_applied[i+1]]
                    segment_length = abs(joint1.x - joint0.x)
                    segment_force = force_per_len * segment_length
                    joint0.update_forces(joint0.fx, joint0.fy + segment_force/2)
                    joint1.update_forces(joint1.fx, joint1.fy + segment_force/2)
                    print(i, segment_force/2, total_len, segment_length, force_per_len, force.fx)

    def print_forces(self):
        for i in range(len(self.joints)):
            print("Joint: {0} Fx={1}, Fy={2}".format(i, self.joints[i].fx, self.joints[i].fy))

    def plot(self):
        for link in self.links:
            plt.plot([self.joints[link[0]].x, self.joints[link[1]].x], [self.joints[link[0]].y, self.joints[link[1]].y], color='black', marker='o')
        for joint in self.joints:
            if(joint.fx is not 0 or joint.fy is not 0):
                plt.arrow(joint.x, joint.y, joint.fx, joint.fy, width=1, label="force")
                plt.annotate('{0}kN'.format(math.sqrt(joint.fx*joint.fx+joint.fy*joint.fy)), xy=(joint.x+joint.fx, joint.y+joint.fy), xytext=(joint.x+joint.fx, joint.y+joint.fy), textcoords='offset points')

        plt.show()


                

    


        