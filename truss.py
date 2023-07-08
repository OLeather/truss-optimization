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
    def __init__(self, type, x, y, fex = 0, fey = 0, grounded=False):
        self.type = type
        self.x = x
        self.y = y
        self.fex = fex
        self.fey = fey
        self.fx = 0
        self.fy = 0
        self.grounded = grounded
    
    def set_fx(self, fx):
        self.fx = fx
    
    def set_fy(self, fy):
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

class Link():
    def __init__(self, i0, i1):
        self.i0 = i0
        self.i1 = i1
        self.member_force = 0 # + for compression, - for tension
        self.multiplier = 1

    def set_member_force(self, force):
        self.member_force = force

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
    
    def solve(self, print_solution = False):
        '''
        @return cost, success : the cost of the bridge and whether the truss survives the load
        '''
        unknowns = []
        unknown_assign = {}
        equations = []
        knowns = []
        for i, joint in enumerate(self.joints):
            # contains tuples of (Link(), other joint index)
            connected_joints = set()

            # Get connected joints 
            for link in self.links:
                if link.i0 == i:
                    connected_joints.add((link, link.i1))
                if link.i1 == i:
                    connected_joints.add((link, link.i0))

            # Calculate Fnet
            fnet_y = {}
            fnet_x = {}
            for connected_joint in connected_joints:
                link = connected_joint[0]
                j = connected_joint[1]

                other = self.joints[j]

                term_name = "F_{0}_{1}".format(i, j)
                term_name_reversed = "F_{0}_{1}".format(j, i)
                if term_name_reversed in unknowns:
                    term_name = term_name_reversed
                else:
                    unknowns.append(term_name)
                    unknown_assign[term_name] = link.set_member_force
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
                    unknown_assign[term_name] = joint.set_fy
        
            if joint.type == JointType.PIN:
                term_name = "N_{0}_x".format(i)
                fnet_x[term_name] = 1
                if term_name not in unknowns:
                    unknowns.append(term_name)
                    unknown_assign[term_name] = joint.set_fx

            # Add fnet_x equation to equations list and external fx to knowns list
            equations.append(fnet_x)
            knowns.append(-joint.fex)

            # Add fnet_y equation to equations list and external fy to knowns list
            equations.append(fnet_y)
            knowns.append(-joint.fey)
            print(knowns)

        A, B = self.construct_system(equations, knowns, unknowns)

        X = np.linalg.inv(A).dot(B)

        for i, unknown in enumerate(unknowns):
            unknown_assign[unknown](X[i][0])
            if print_solution:
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
        return success

    def calculate_cost(self):
        cost = 0

        # Calculate cost of trusses ($15 per m)
        for i, link in enumerate(self.links):
            j0 = self.joints[link.i0]
            j1 = self.joints[link.i1]
            d = math.sqrt(math.pow(j1.x-j0.x,2) + math.pow(j1.y-j0.y, 2))
            cost += d * self.COST_PER_M
        
        # Calculate cost of gussets ($5 per joint)
        cost += len(self.joints) * self.COST_PER_JOINT

        return cost

    def apply_external_forces(self):
        for force in self.forces:
            if len(force.joints_applied) == 1:
                self.joints[force.joints_applied[0]].fex = force.fex
                self.joints[force.joints_applied[0]].fey = force.fey
            else:    
                total_len = abs(self.joints[force.joints_applied[-1]].x - self.joints[force.joints_applied[0]].x)
                force_per_len = float(force.fy) / total_len
                for i in range(len(force.joints_applied)-1):
                    joint0 = self.joints[force.joints_applied[i]]
                    joint1 = self.joints[force.joints_applied[i+1]]
                    segment_length = abs(joint1.x - joint0.x)
                    segment_force = force_per_len * segment_length
                    joint0.fex = joint0.fex
                    joint0.fey = joint0.fey + segment_force/2
                    joint1.fex = joint1.fex
                    joint1.fey = joint1.fey + segment_force/2

    def plot(self, title = "", plot_external = False, plot_member = False):
        for link in self.links:
            plt.plot([self.joints[link.i0].x, self.joints[link.i1].x], [self.joints[link.i0].y, self.joints[link.i1].y], color='black', marker='o')
            plt.annotate('{0}'.format(link.i0), xy=(self.joints[link.i0].x, self.joints[link.i0].y), xytext=(2, 2), textcoords='offset points')
            plt.annotate('{0}'.format(link.i1), xy=(self.joints[link.i1].x, self.joints[link.i1].y), xytext=(2, 2), textcoords='offset points')

        for joint in self.joints:
            if((joint.fex != 0 or joint.fey != 0)) and plot_external:
                mag = round(math.sqrt(joint.fex*joint.fex+joint.fey*joint.fey), 3)
                plt.arrow(joint.x, joint.y, joint.fex, joint.fey, width=.5, label="force")
                # plt.annotate('{0}kN'.format(str(mag)), xy=(joint.x+joint.fex, joint.y+joint.fey), xytext=(0, 0), textcoords='offset points')
                t = plt.text(joint.x+joint.fex, joint.y+joint.fey, '{0}kN'.format(mag), fontsize=10)
                t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='grey'))
            if((joint.fx != 0 or joint.fy != 0)) and plot_member:
                mag = round(math.sqrt(joint.fx*joint.fx+joint.fy*joint.fy), 3)
                plt.arrow(joint.x, joint.y, joint.fx, joint.fy, width=.5, label="force")
                t = plt.text(joint.x+joint.fx, joint.y+joint.fy, '{0}kN'.format(mag), fontsize=10)
                t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='grey'))
        
        if plot_member:
            for link in self.links:
                joint0 = self.joints[link.i0]
                joint1 = self.joints[link.i1]

                mx = (joint1.x+joint0.x)/2
                my = (joint1.y+joint0.y)/2

                force = round(link.member_force, 3)
                color = 'blue' if force > 0 else 'red'
                t = plt.text(mx, my, '{0}kN'.format(force), fontsize=10)
                t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))

        plt.title(title)
        plt.show()


                

    


        