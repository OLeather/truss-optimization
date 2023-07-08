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

    def __init__(self, joints, links, forces, multipliers):
        '''
        @param joints: list of Joint objects
        @param links: list of tuples connecting joints by index
        '''
        self.joints = joints
        self.links = links
        self.forces = forces
        self.multipliers = multipliers
    
    def solve(self, print_solution = False):
        '''
        @return cost, success : the cost of the bridge and whether the truss survives the load
        '''
        unknowns = []
        equations = []
        knowns = []
        for i, joint in enumerate(self.joints):
            connected_joints = set()

            # Get connected joints 
            for link in self.links:
                if link[0] == i:
                    connected_joints.add(link[1])
                if link[1] == i:
                    connected_joints.add(link[0])

            # Calculate Fnet
            fnet_y = {}
            fnet_x = {}
            for j in connected_joints:
                other = self.joints[j]

                term_name = "F_{0}_{1}".format(i, j)
                term_name_reversed = "F_{0}_{1}".format(j, i)
                if term_name_reversed in unknowns:
                    term_name = term_name_reversed
                else:
                    unknowns.append(term_name)
                
                vx = (other.x-joint.x)
                vy = (other.y-joint.y)
                ux = vx / math.sqrt(vx*vx+vy*vy)
                uy = vy / math.sqrt(vx*vx+vy*vy)

                fnet_x[term_name] = ux
                fnet_y[term_name] = uy
            
            if joint.type == JointType.ROLLER or joint.type == JointType.PIN:
                term_name = "N_{0}_y".format(i)
                fnet_y[term_name] = 1
                if term_name not in unknowns:
                    unknowns.append(term_name)
        
            if joint.type == JointType.PIN:
                term_name = "N_{0}_x".format(i)
                fnet_x[term_name] = 1
                if term_name not in unknowns:
                    unknowns.append(term_name)
            
            # Add fnet_x equation to equations list and external fx to knowns list
            equations.append(fnet_x)
            knowns.append(-joint.fx)

            # Add fnet_y equation to equations list and external fy to knowns list
            equations.append(fnet_y)
            knowns.append(-joint.fy)

        A, B = self.construct_system(equations, knowns, unknowns)

        X = np.linalg.inv(A).dot(B)

        if print_solution:
            for i, unknown in enumerate(unknowns):
                print(unknown, "=", X[i][0])

        # Calculate cost and success
        success = self.check_constraints(X, unknowns)        
        cost = self.calculate_cost()

        if not success:
            cost += 10000
        return cost, success
    
    def construct_system(self, equations, knowns, unknowns):
        A = []
        B = []
        
        for i, equation in enumerate(equations):
            row = []
            for unknown in unknowns:
                if unknown in equation:
                    row.append(equation[unknown])
                else:
                    row.append(0)
            A.append(row)
            B.append([knowns[i]])

        return A, B

    def check_constraints(self, X, unknowns):
        success = True
        for i, num in enumerate(X):
            term = unknowns[i]
            split = term.split('_')
            # Only apply constraints to member forces
            if split[0] == 'F':
                i1 = int(split[1])
                i2 = int(split[2])
                tup = (i1, i2)
                tup_inv = (i2, i1)

                if tup in self.links:
                    j = self.links.index(tup)
                    multiplier = self.multipliers[j]
                if tup_inv in self.links:
                    j = self.links.index(tup_inv)
                    multiplier = self.multipliers[j]

                # print(multiplier)
                
                # Compressive
                if num > 0 and abs(num) > self.MAX_COMPRESSIVE * multiplier:
                    success = False

                # Tensile
                if num < 0 and abs(num) > self.MAX_TENSILE * multiplier:
                    success = False
        return success

    def calculate_cost(self):
        cost = 0

        # Calculate cost of trusses ($15 per m)
        for i, link in enumerate(self.links):
            multiplier = self.multipliers[i]
            j0 = self.joints[link[0]]
            j1 = self.joints[link[1]]
            d = math.sqrt(math.pow(j1.x-j0.x,2) + math.pow(j1.y-j0.y, 2))
            cost += d * self.COST_PER_M * multiplier
        
        # Calculate cost of gussets ($5 per joint)
        cost += len(self.joints) * self.COST_PER_JOINT

        return cost

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


                

    


        