from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np

# class syntax
class JointType(Enum):
    PIN = 1
    ROLLER = 2

# functional syntax
JointType = Enum('JointType', ['PIN', 'ROLLER'])

class Joint():
    def __init__(self, type, x, y, fx = 0, fy = 0, grounded=False):
        self.type = type
        self.x = x
        self.y = y
        self.update_forces(fx, fy)
        self.grounded = grounded
    
    def update_forces(self, fx, fy):
        self.fx = fx
        self.fy = fy

class Force():
    def __init__(self, magnitude, direction, joints_applied):
        self.x = magnitude*math.cos(direction)
        self.y = magnitude*math.sin(direction)
        self.joints_applied = joints_applied
        self.compressive = 0
        self.tensile = 0
    
    def __init__(self, fx, fy, joints_applied):
        self.fx = fx
        self.fy = fy
        self.joints_applied = joints_applied
        self.compressive = 0
        self.tensile = 0

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
        unknowns = {}

        for i in range(len(self.joints)):
            joint = self.joints[i]

            connected_joints = set()
            
            fknown_x = joint.fx
            fknown_y = joint.fy

            for link in self.links:
                if link[0] == i:
                    connected_joints.add(link[1])
                if link[1] == i:
                    connected_joints.add(link[0])

            x_terms = []
            y_terms = []

            for connected_joint_i in connected_joints:
                connected_joint = self.joints[connected_joint_i]
                vx = connected_joint.x-joint.x
                vy = connected_joint.y-joint.y
                ux = vx/math.sqrt(vx*vx+vy*vy)
                uy = vy/math.sqrt(vx*vx+vy*vy)
                
                x_terms.append(("f_{0}{1}".format(i, connected_joint_i), ux))
                y_terms.append(("f_{0}{1}".format(i, connected_joint_i), uy))

            format_str_x = "F_net_{0}x = 0 = ".format(i) + str(fknown_x) + ((" + " + "F_N{0}x".format(i)) if joint.grounded else "")
            for term in x_terms:
                format_str_x += " + " + str(term[0]) + "*{0}".format(term[1])
            format_str_y = "F_net_{0}y = 0 = ".format(i) + str(fknown_y) + ((" + " + "F_N{0}y".format(i)) if joint.type == JointType.PIN and joint.grounded else "")
            for term in y_terms:
                format_str_y += " + " + str(term[0]) + "*{0}".format(term[1])
            
            print(format_str_x)
            print(format_str_y)
            
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

    def print_forces(self):
        for i in range(len(self.joints)):
            print("Joint: {0} Fx={1}, Fy={2}".format(i, self.joints[i].fx, self.joints[i].fy))

    def plot(self):
        for link in self.links:
            plt.plot([self.joints[link[0]].x, self.joints[link[1]].x], [self.joints[link[0]].y, self.joints[link[1]].y], color='black', marker='o')
            plt.annotate('{0}'.format(link[0]), xy=(self.joints[link[0]].x, self.joints[link[0]].y), xytext=(self.joints[link[0]].x, self.joints[link[0]].y), textcoords='offset points')
            plt.annotate('{0}'.format(link[1]), xy=(self.joints[link[1]].x, self.joints[link[1]].y), xytext=(self.joints[link[1]].x, self.joints[link[1]].y), textcoords='offset points')

        for joint in self.joints:
            if(joint.fx != 0 or joint.fy != 0):
                plt.arrow(joint.x, joint.y, joint.fx, joint.fy, width=1, label="force")
                plt.annotate('{0}kN'.format(math.sqrt(joint.fx*joint.fx+joint.fy*joint.fy)), xy=(joint.x+joint.fx, joint.y+joint.fy), xytext=(joint.x+joint.fx, joint.y+joint.fy), textcoords='offset points')

        plt.show()


                

    


        