from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import json


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
        self.distance = 0

    def set_member_force(self, force):
        self.member_force = force



def length(x0, y0, x1, y1):
    return sp.sqrt(sp.Pow(x1-x0,2)+sp.Pow(y1-y0,2))
    
class Truss():
    MAX_TENSILE = 9 # kN
    MAX_COMPRESSIVE = 6 # kN
    COST_PER_JOINT = 5 # Dollars
    COST_PER_M = 15 # Dollars
    MAX_MULTIPLIER = 3 # Triple joint
    MINIMUM_TRUSS_LENGTH = 1

    def __init__(self, joints = [], links = [], forces = [], filename = ""):
        '''
        @param joints: list of Joint objects
        @param links: list of tuples connecting joints by index
        '''
        self.joints = joints
        self.links = links
        self.forces = forces
        if filename != "":
            self.open_file(filename, print_data = True)

    
    def open_file(self, filename, print_data=False):
        f = open(filename)
  
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        self.joints = []
        self.links = []
        self.forces = []
        for i, node in enumerate(data["nodes"]):
            split = node.split(',')
            x = float(split[0])
            y = float(split[1])
            type = JointType.GUSSET
            grounded = False
            for support in data["supports"]:
                attached = int(support)
                if attached == i:
                    type_str = str(data["supports"][support])
                    if type_str == "P":
                        type = JointType.PIN
                        grounded = True
                    if type_str == "Rh":
                        type = JointType.ROLLER
                        grounded = True
            if print_data:
                print("Joint",i, type, x, y, grounded)
            self.joints.append(Joint(type, x, y, grounded=grounded))

        for member in data["members"]:
            split = member.split(',')
            i0 = int(split[0])
            i1 = int(split[1])
            self.links.append(Link(i0, i1))
            if print_data:
                print("Link:", i0, i1)

        for force in data["forces"]:
            split = force.split(',')
            i = int(split[0])
            fx = float(split[1])
            fy = float(split[2])
            if print_data:
                print("Force", i, fx, fy)
        
            self.forces.append(Force(fx=fx, fy=fy, joints_applied=[i]))

        print("joints = [")
        for joint in self.joints:
            print("\tJoint({0}, {1}, {2}, grounded={3}),".format(("JointType.PIN" if joint.type == JointType.PIN else ("JointType.ROLLER" if joint.type == JointType.ROLLER else "JointType.GUSSET")), joint.x, joint.y, joint.grounded))
        print("]")

        print("links = [")
        for link in self.links:
            print("\tLink({0}, {1}),".format(link.i0, link.i1))
        print("]")

        print("forces = [")
        for force in self.forces:
            print("\tForce({0}, {1}, joints_applied=[{2}]),".format(force.fx, force.fy, force.joints_applied[0]))
        print("]")

        print("joints = [")
        for i, joint in enumerate(self.joints):
            if joint.type == JointType.PIN:
                print("\tJoint(JointType.PIN, {0}, {1}, grounded={2}),".format(joint.x, joint.y, joint.grounded))
            elif joint.type == JointType.ROLLER:
                print("\tJoint(JointType.ROLLER, {0}, {1}, grounded={2}),".format(joint.x, joint.y, joint.grounded))
            else:
                if joint.y == 0:
                    print("\tJoint(JointType.GUSSET, {0}, {1}, grounded={2}),".format(joint.x, joint.y, joint.grounded))
                else:
                    print("\tJoint(JointType.GUSSET, xs[{0}], ys[{1}], grounded={2}),".format(i, i, joint.grounded))
        print("]")

        print("x0 = [")
        for joint in self.joints:
            print("\t{0},".format(joint.x))
        for joint in self.joints:
            print("\t{0},".format(joint.y))
        print("]")

    
    def print_joints(self):
        print("joints = [")
        for i, joint in enumerate(self.joints):
            print("\tJoint({0}, {1}, {2}, grounded={3}), # {4}".format(("JointType.PIN" if joint.type == JointType.PIN else ("JointType.ROLLER" if joint.type == JointType.ROLLER else "JointType.GUSSET")), joint.x, joint.y, joint.grounded, i))
        print("]")

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
            # print(knowns)

        A, B = self.construct_system(equations, knowns, unknowns)

        X = []
        try:
            X = np.linalg.inv(A).dot(B)
        except:
            X = [[10000]]*len(unknowns)
            print("Unsolvable")
            
        for i, unknown in enumerate(unknowns):
            unknown_assign[unknown](X[i][0])
            if print_solution:
                print(unknown, "=", X[i][0])
        
        # Calculate cost and success
        success = self.check_constraints()        
        cost = self.calculate_cost()

        if not success:
            cost += 10000
        return cost, success

    def generate_symbolic_equations(self):
        xs = []
        ys = []

        for i, joint in enumerate(self.joints):
            xi = sp.symbols('x{0}'.format(i))
            yi = sp.symbols('y{0}'.format(i))
            
            xs.append(xi)
            ys.append(yi)
        
        for j, link in enumerate(self.links):
            link.distance = sp.symbols('d{0}'.format(j))
        
        equations = []
        forces = []
        distances = []
        distance_joints = []
        unknown_assign = {}
        for i, joint in enumerate(self.joints):
            # contains tuples of (Link(), other joint index)
            connected_joints = set()


            # Get connected joints
            for link in self.links:
                if link.i0 == i:
                    connected_joints.add((link, link.i1))
                if link.i1 == i:
                    connected_joints.add((link, link.i0))

            fnetX = joint.fex
            fnetY = joint.fey

            # Add member forces to equation
            for k, (link, j) in enumerate(connected_joints):
                xi = xs[i]
                xj = xs[j]
                yi = ys[i]
                yj = ys[j]
                
                term_name = "F{0}{1}".format(i, j)
                term_name_reversed = "F{0}{1}".format(j, i)

                force = sp.symbols(term_name)
                force_reversed = sp.symbols(term_name_reversed)
                if force_reversed in forces:
                    force = force_reversed
                else:
                    forces.append(force)
                    unknown_assign[force] = link.set_member_force

                vx = xj-xi
                vy = yj-yi
                # d = sp.sqrt(sp.Pow(vx,2) + sp.Pow(vy,2))
                    
                ux = vx/link.distance
                uy = vy/link.distance
                
                fnetX += ux * force
                fnetY += uy * force

            # Add normal forces of supports to equations
            if joint.type == JointType.ROLLER or joint.type == JointType.PIN:
                term_name = "N{0}y".format(i)
                force = sp.symbols(term_name)
                fnetY += force
                if force not in forces:
                    forces.append(force)
                    unknown_assign[force] = joint.set_fy

            if joint.type == JointType.PIN:
                term_name = "N{0}x".format(i)
                force = sp.symbols(term_name)
                fnetX += force
                if force not in forces:
                    forces.append(force)
                    unknown_assign[force] = joint.set_fx
            
            equations.append(fnetX)
            equations.append(fnetY)
        
        solution = sp.solve(equations, forces, dict=True)[0]

        
        for force, eqn in solution.items():
        #     eqn_new = eqn
        #     for i, distance in enumerate(distances):
        #         joints = distance_joints[i]
        #         eqn_new = eqn_new.subs(distance, length(joints[0], joints[1], joints[2], joints[3]))
        #     solution[force] = eqn_new
            solution[force] = sp.simplify(eqn)
            unknown_assign[force](eqn)

        cost = self.compute_cost_function(self.links)

        return solution, xs + ys, cost
    
    
    
    def compute_cost_function(self, links):
        # The cost is computed as the following:
        # Sum of the length of each member * (force of member * multiplier function)
        
        def numeric_multiplier(x):
            # Tensile
            if x < 0:
                return 1 if abs(x) < self.MAX_TENSILE else (2 if abs(x) < self.MAX_TENSILE * 2 else 3)
            else:
                return 1 if abs(x) < self.MAX_COMPRESSIVE else (2 if abs(x) < self.MAX_COMPRESSIVE * 2 else 3)

        def multiplier(x):        
            e = 2.71828
            b = 23
            n = 8.885
            m = 17.885
            p = 10
            j = 5.885
            k = 11.885
            return 1 + sp.Pow(1/(1+sp.Pow(e,-b*(-x-n))), p) + sp.Pow(1/(1+sp.Pow(e,-b*(-x-m))), p) + sp.Pow(1/(1+sp.Pow(e,-b*(x-j))), p) + sp.Pow(1/(1+sp.Pow(e,-b*(x-k))), p)

        cost = len(self.joints) * self.COST_PER_JOINT

        for link in links:
            cost += multiplier(link.member_force) * length(self.joints[link.i0].x, self.joints[link.i0].y, self.joints[link.i1].x, self.joints[link.i1].y) * self.COST_PER_M
        
        

        return cost
    
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

    def check_constraints(self):
        success = True
        for link in self.links:
            abs_max = 0

            # Tensile
            if link.member_force > 0:
                abs_max = self.MAX_TENSILE
            # Compressive
            else:
                abs_max = self.MAX_COMPRESSIVE

            for i in range(1, self.MAX_MULTIPLIER+1):
                link.multiplier = i

                if abs(link.member_force) < abs_max * link.multiplier:
                    break
                elif i == self.MAX_MULTIPLIER:
                    success = False

            # Check for minimum distance of 1 m
            j0 = self.joints[link.i0]
            j1 = self.joints[link.i1]
            d = math.sqrt(math.pow(j1.x-j0.x,2) + math.pow(j1.y-j0.y, 2))
            if d < self.MINIMUM_TRUSS_LENGTH:
                # print(d)
                success = False
            
            
        return success

    def calculate_cost(self):
        cost = 0
        total_m = 0
        # Calculate cost of trusses ($15 per m)
        for i, link in enumerate(self.links):
            j0 = self.joints[link.i0]
            j1 = self.joints[link.i1]
            d = math.sqrt(math.pow(j1.x-j0.x,2) + math.pow(j1.y-j0.y, 2))
            cost += d * self.COST_PER_M * link.multiplier
            total_m += d * link.multiplier
        
        # Calculate cost of gussets ($5 per joint)
        cost += len(self.joints) * self.COST_PER_JOINT

        return cost

    def apply_external_forces(self):
        for force in self.forces:
            if len(force.joints_applied) == 1:
                self.joints[force.joints_applied[0]].fex = force.fx
                self.joints[force.joints_applied[0]].fey = force.fy
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
            plt.plot([self.joints[link.i0].x, self.joints[link.i1].x], [self.joints[link.i0].y, self.joints[link.i1].y], color='black', marker='o', linewidth=link.multiplier)
            plt.annotate('{0}'.format(link.i0), xy=(self.joints[link.i0].x, self.joints[link.i0].y), xytext=(2, 2), textcoords='offset points')
            plt.annotate('{0}'.format(link.i1), xy=(self.joints[link.i1].x, self.joints[link.i1].y), xytext=(2, 2), textcoords='offset points')

        for joint in self.joints:
            if((joint.fex != 0 or joint.fey != 0)) and plot_external:
                mag = round(math.sqrt(joint.fex*joint.fex+joint.fey*joint.fey), 3)
                plt.arrow(joint.x, joint.y, joint.fex/6, joint.fey/6, width=.3, label="force")
                # plt.annotate('{0}kN'.format(str(mag)), xy=(joint.x+joint.fex, joint.y+joint.fey), xytext=(0, 0), textcoords='offset points')
                t = plt.text(joint.x+joint.fex/6, joint.y+joint.fey/6, '{0}kN'.format(mag), fontsize=10)
                t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='grey'))
            if((joint.fx != 0 or joint.fy != 0)) and plot_member:
                mag = round(math.sqrt(joint.fx*joint.fx+joint.fy*joint.fy), 3)
                plt.arrow(joint.x, joint.y, joint.fx/6, joint.fy/6, width=.3, label="force")
                t = plt.text(joint.x+joint.fx/6, joint.y+joint.fy/6, '{0}kN'.format(mag), fontsize=10)
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


                

    


        