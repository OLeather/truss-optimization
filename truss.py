from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy

# class syntax
class JointType(Enum):
    PIN = 1
    ROLLER = 2
    GUSSET = 3

# functional syntax
JointType = Enum('JointType', ['PIN', 'ROLLER', 'GUSSET'])

class Joint():
    def __init__(self, type, x, y, fx = 0, fy = 0, grounded=False):
        self.type = type
        self.x = x
        self.y = y
        self.update_forces(fx, fy)
        self.grounded = grounded
        self.fxs = []
        self.fys = []
    
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
    MAX_TENSILE = 50 # kN
    MAX_COMPRESSIVE = 50 # kN
    COST_PER_JOINT = 5 # Dollars
    COST_PER_M = 15 # Dollars

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
        @return cost, success : the cost of the bridge and whether the truss survives the load
        '''
        unknowns = []
        equations = []
        knowns = []
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
    
            # Moment equation at pin joint
            if(joint.type == JointType.PIN):
                known = 0
                moment_eqn = {}
                # Sum up the known moments
                for j in range(len(self.joints)):
                    if j != i:
                        other = self.joints[j]
                        d = abs(other.x - joint.x)
                        # Subtract the left side of the equation by fy * d for each external force
                        known -= other.fy * d
                        if other.grounded:
                            term_name = "N_{0}_y".format(j)
                            unknowns.append(term_name)
                            # The unknown scalar = distance from joint to other
                            moment_eqn[term_name] = d

                equations.append(moment_eqn)
                knowns.append(known)

            # Fnet Equations For Connected Joints
            fnet_x = {}
            fnet_y = {}
            for connected_joint_i in connected_joints:
                connected_joint = self.joints[connected_joint_i]
                
                term_name = "F_{0}_{1}".format(i, connected_joint_i)
                term_name_reverse = "F_{0}_{1}".format(connected_joint_i, i)
                
                reversed = 1
                if term_name_reverse in unknowns:
                    term_name = term_name_reverse
                    reversed = -1
                else:
                    unknowns.append(term_name)
                
                vx = connected_joint.x-joint.x
                vy = connected_joint.y-joint.y
                ux = vx/math.sqrt(vx*vx+vy*vy) * reversed
                uy = vy/math.sqrt(vx*vx+vy*vy) * reversed

                fnet_x[term_name] = ux
                fnet_y[term_name] = uy

            if(joint.type == JointType.ROLLER or joint.type == JointType.PIN):
                term_name = "N_{0}_y".format(i)
                fnet_y[term_name] = 1
                if term_name not in unknowns:
                    unknowns.append(term_name)            
            if(joint.type == JointType.PIN):
                term_name = "N_{0}_x".format(i)
                fnet_x[term_name] = 1
                if term_name not in unknowns:
                    unknowns.append(term_name)

                
            equations.append(fnet_x)
            knowns.append(-joint.fx)
            equations.append(fnet_y)
            knowns.append(-joint.fy)

        A = []
        for eqn in equations:
            row = []
            for unknown in unknowns:
                if unknown in eqn:
                    row.append(eqn[unknown])
                else:
                    row.append(0)
            A.append(row)

        B = []
        for known in knowns:
            B.append([known])

        _, indices = sympy.Matrix(A).rref()
 
        A_ = []
        B_ = []
        for i in indices:
            A_.append(A[i])
            B_.append(B[i])
        X = np.linalg.inv(A_).dot(B_)
    
        # pretty print equations
        for i in range(len(A_)):
            eqn = A_[i]
            left_side = str(round(B_[i][0],3)) + " = "
            right_side = ""
            for j in range(len(eqn)):      
                if(eqn[j] != 0):         
                    key = unknowns[j]
                    split = key.split('_')
                    key = split[0] + "_{" + split[1] + split[2] + "}"
                    if right_side != "":
                        right_side += " + "
                    right_side += str(round(eqn[j],3)) + str(key)

            eqn_str = "" + left_side + right_side + "\\" + "\\"
            
            # print(eqn_str)
        
        for i in range(len(X)):
            solved = X[i][0]
            key = unknowns[i]
            split = key.split('_')
            key = split[0] + "_{" + split[1] + split[2] + "}"
            # print(key + " = " + str(round(solved,3)), "\\\\")

        i = 0
        for unknown in unknowns:
            split = unknown.split('_')

            if split[0] == 'N':
                joint0 = self.joints[int(split[1])]
                if split[2] == 'x':
                    joint0.fxs.append(X[i][0])
                    joint0.fys.append(0)
                if split[2] == 'y':
                    joint0.fxs.append(0)
                    joint0.fys.append(X[i][0])
            if split[0] == 'F':
                force = X[i][0]
                joint0 = self.joints[int(split[1])]
                joint1 = self.joints[int(split[2])]
                vx = joint1.x-joint0.x
                vy = joint1.y-joint0.y
                ux = vx/math.sqrt(vx*vx+vy*vy)
                uy = vy/math.sqrt(vx*vx+vy*vy)

                # print(unknown, split, force*ux, force*uy, -force*ux, -force*uy)

                joint0.fxs.append(force*ux)
                joint0.fys.append(force*uy)
                joint1.fxs.append(-force*ux)
                joint1.fys.append(-force*uy)
            i += 1

        success = True
        for joint in self.joints:
            for i in range(len(joint.fxs)):
                fx = joint.fxs[i]
                fy = joint.fys[i]
                magnitude = math.sqrt(fx*fx+fy*fy)
                if magnitude > self.MAX_TENSILE:
                    success = False
        
        cost = 0
        cost += len(self.joints) * self.COST_PER_JOINT

        for link in self.links:
            joint0 = self.joints[link[0]]
            joint1 = self.joints[link[1]]
            distance = math.sqrt((joint0.x-joint1.x)*(joint0.x-joint1.x)+(joint0.y-joint1.y)*(joint0.y-joint1.y))
            cost += distance * self.COST_PER_M

        if not success:
            cost += 100000
        return cost, success

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

    def plot(self, title = "", plot_external = False, plot_member = False):
        for link in self.links:
            plt.plot([self.joints[link[0]].x, self.joints[link[1]].x], [self.joints[link[0]].y, self.joints[link[1]].y], color='black', marker='o')
            plt.annotate('{0}'.format(link[0]), xy=(self.joints[link[0]].x, self.joints[link[0]].y), xytext=(2, 2), textcoords='offset points')
            plt.annotate('{0}'.format(link[1]), xy=(self.joints[link[1]].x, self.joints[link[1]].y), xytext=(2, 2), textcoords='offset points')

        for joint in self.joints:
            if((joint.fx != 0 or joint.fy != 0) and plot_external):
                # print(joint.x, joint.y, joint.fx, joint.fy)
                plt.arrow(joint.x, joint.y, joint.fx, joint.fy, width=1, label="force")
                plt.annotate('{0}kN'.format(str(round(math.sqrt(joint.fx*joint.fx+joint.fy*joint.fy), 3))), xy=(joint.x+joint.fx, joint.y+joint.fy), xytext=(0, 0), textcoords='offset points')
            if plot_member:
                for i in range(len(joint.fxs)):
                    fx = joint.fxs[i]
                    fy = joint.fys[i]
                    if fx != 0 or fy != 0:
                        plt.arrow(joint.x, joint.y, fx, fy, width=.5, label="force")
                        plt.annotate('{0}kN'.format(str(round(math.sqrt(fx*fx+fy*fy), 3))), xy=(joint.x+fx, joint.y+fy), xytext=(0, 0), textcoords='offset points')
        plt.title(title)
        plt.show()


                

    


        