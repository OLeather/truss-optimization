import casadi
from casadi import sqrt, constpow, vertcat, horzcat, SX, MX, solve, jacobian, hessian, Function, fabs
from truss import JointType
from sortedcollections import OrderedSet
import json
import matplotlib.pyplot as plt
import math

class Joint():
    def __init__(self, type):
        self.type = type
        self.fex = 0
        self.fey = 0
        self.fx = 0
        self.fy = 0
    def set_fx(self, fx, jac, hess):
        self.fx = fx
    
    def set_fy(self, fy, jac, hess):
        self.fy = fy

class Link():
    def __init__(self, i0, i1):
        self.i0 = i0
        self.i1 = i1
        self.force = 0
    
    def set_force(self, f, jac, hess):
        self.force = f
        self.force_jac = jac
        self.force_hess = hess

    def set_length(self, l, jac, hess):
        self.length = l
        self.length_jac = jac
        self.length_hess = hess

class Force():
    def __init__(self, fx, fy, joints):
        self.fx = fx
        self.fy = fy
        self.joints = joints

def length(x0, y0, x1, y1):
    return sqrt(constpow(x0-x1,2) + constpow(y0-y1,2))


def open_file(filename, y_shift = 0, flip=False, print_data=False):
        f = open(filename)
        flip_ = 1
        if(flip):
            flip_ = -1
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        joints = []
        links = []
        forces = []
        xs = [0] * len(data["nodes"]*2)
        y_offset = len(data["nodes"])
        bridge_joints = []
        x_offset = 0
        for i, node in enumerate(data["nodes"]):
            split = node.split(',')
            x = float(split[0])
            y = float(split[1])
            type = JointType.GUSSET
            for support in data["supports"]:
                attached = int(support)
                if attached == i:
                    type_str = str(data["supports"][support])
                    if type_str == "P":
                        type = JointType.PIN
                    if type_str == "Rh":
                        type = JointType.ROLLER
            if print_data:
                print("Joint", i, type, x, y)
            x_offset = min(x, x_offset)
            xs[i] = x
            xs[i+y_offset] = y*flip_ + y_shift
            if(y*flip_ + y_shift == 0):
                bridge_joints.append(i)
            joints.append(Joint(type))
        for i in range(int(len(xs)/2)):
            xs[i] = xs[i] - x_offset
        bridge_joints.sort(key=lambda i : xs[i])

        for member in data["members"]:
            split = member.split(',')
            i0 = int(split[0])
            i1 = int(split[1])
            links.append(Link(i0, i1))
            if print_data:
                print("Link:", i0, i1)

        for force in data["forces"]:
            split = force.split(',')
            i = int(split[0])
            fx = float(split[1])
            fy = float(split[2])
            if print_data:
                print("Force", i, fx, fy)
        
            forces.append(Force(fx=fx, fy=-fy, joints=[i]))

        print(xs)
        return Truss(joints, links, [Force(fx=0, fy=2.5, joints=bridge_joints)]), bridge_joints, xs

        
class Truss():
    FORCE_PER_M = 5 # kg/m
    COST_PER_JOINT = 5 # Dollars
    COST_PER_M = 15 # Dollars

    def __init__(self, joints, links, forces):
        self.joints = joints
        self.links = links
        self.forces = forces
        self.y_offset = len(joints)
        self.xs = SX.sym('xs', len(joints)*2)
        # self.ys = SX.sym('ys', len(joints))
        self.apply_link_lengths()
        self.apply_external_forces()
        self.construct_knowns_vector()
        self.A = self.construct_equations_matrix()
        self.cost, self.real_cost = self.compute_cost_function()
        # self.jacobian = self.compute_jacobian()
        # self.hessian = self.compute_hessian()

    def apply_link_lengths(self):
        for i, link in enumerate(self.links):
            l = length(self.xs[link.i0], self.xs[link.i0 + self.y_offset], self.xs[link.i1], self.xs[link.i1 + self.y_offset])
            jac = jacobian(l, self.xs)
            [hess, g] = hessian(l, self.xs)
            link.set_length(l, jac, hess)
            link.force = SX.sym('F{0}'.format(i))

    def apply_external_forces(self):
        for force in self.forces:
            # print(force.joints)
            if len(force.joints) == 1:
                self.joints[force.joints[0]].fex = force.fx
                self.joints[force.joints[0]].fey = force.fy
            elif len(force.joints) > 1:
                for i in range(len(force.joints)-1):
                    j = force.joints[i]
                    k = force.joints[i+1]

                    self.joints[j].fex += force.fx * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2
                    self.joints[j].fey += force.fy * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2 
                    self.joints[k].fex += force.fx * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2
                    self.joints[k].fey += force.fy * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2

        #             print(j, k, self.joints[i].fex, self.joints[i].fey) 
        # for i in range(len(self.joints)):
        #     print(i, self.joints[i].fey)

    def construct_knowns_vector(self):
        knowns = []
        for joint in self.joints:
            knowns.append(joint.fex)
            knowns.append(joint.fey)
        self.external_forces = vertcat(*knowns)

    def construct_equations_matrix(self):
        equations = []
        forces = OrderedSet()
        equation_names = []
        unknown_assign = {}
        for i, joint in enumerate(self.joints):
            connected_joints = OrderedSet()

            # Get connected joints
            for j, link in enumerate(self.links):
                if link.i0 == i:
                    connected_joints.add((j, link, link.i1))
                if link.i1 == i:
                    connected_joints.add((j, link, link.i0))

            fnetX = {}
            fnetY = {}

            # Add member forces to equation
            for k, (h, link, j) in enumerate(connected_joints):
                xi = self.xs[i]
                xj = self.xs[j]
                yi = self.xs[i + self.y_offset]
                yj = self.xs[j + self.y_offset]
                
                vx = xj-xi
                vy = yj-yi
                # d = sp.sqrt(sp.Pow(vx,2) + sp.Pow(vy,2))
                    
                ux = vx/link.length
                uy = vy/link.length
                
                fnetX[link.force] = ux
                fnetY[link.force] = uy
                unknown_assign[link.force] = link.set_force
                forces.add(link.force)
                
                # forces['F{0}'.format(h)] = link.force
                
            # print(fnetX)
            # print(fnetY)

            # Add normal forces of supports to equations
            if joint.type == JointType.ROLLER or joint.type == JointType.PIN:
                force = SX.sym('N{0}y'.format(i))
                fnetY[force] = 1
                forces.add(force)
                unknown_assign[force] = joint.set_fy
                # if force not in forces:
                #     forces.append(force)

            if joint.type == JointType.PIN:
                force = SX.sym('N{0}x'.format(i))
                fnetX[force] = 1
                forces.add(force)
                unknown_assign[force] = joint.set_fx
                # if force not in forces:
                #     forces.append(force)
            
            equations.append(fnetX)
            equation_names.append('Fnetx{0}'.format(i))
            equations.append(fnetY)
            equation_names.append('Fnety{0}'.format(i))

        A = []
        for i, equation in enumerate(equations):
            eqn = []
            for force in forces:
                if force in equation:
                    eqn.append(equation[force])
                else:
                    eqn.append(0)
            A.append(eqn)
            # print(equation_names[i])
        # print(A)
        As = []
        for row in A:
            As.append(horzcat(*row))
        
        Ab = vertcat(*As)
        self.Ab_ = Ab
        # print(Ab)
        solution = solve(Ab, self.external_forces)
        
        # print(solution.shape)
        for i, assign in enumerate(unknown_assign.values()):
            f = solution[i]
            # jac = jacobian(solution[i], self.xs)
            # [hes, g] = hessian(solution[i], self.xs)
            assign(f, None, None)
        
        return solution


    def compute_cost_function(self):
        # https://www.desmos.com/calculator/40bxe1edsw
        def multiplier(x):
            e = 2.71828
            p = 10
            b = 10
            # n = 8.7
            # m = 17.7
            # j = 5.7
            # k = 11.7
            n = 8.885
            m = 17.885
            j = 5.885
            k = 11.885
            return constpow(1/(1+constpow(e,-b*(x-n))), p) \
                + constpow(1/(1+constpow(e,-b*(x-m))), p) \
                + constpow(1/(1+constpow(e,-b*(-x-j))), p) \
                + constpow(1/(1+constpow(e,-b*(-x-k))), p) + 1
    
        def real_multiplier(x):
            relu_neg = (casadi.sign(x)-1)/2*x
            relu_pos = (casadi.sign(x)+1)/2*x
            x_ = casadi.ceil(casadi.fabs(relu_neg)/6) + casadi.ceil(casadi.fabs(relu_pos)/9)
            return x_
        
        cost = len(self.joints) * self.COST_PER_JOINT
        real_cost = len(self.joints) * self.COST_PER_JOINT
        for link in self.links:
            cost += multiplier(link.force) * length(self.xs[link.i0], self.xs[link.i0 + self.y_offset], self.xs[link.i1], self.xs[link.i1 + self.y_offset]) * self.COST_PER_M
            real_cost += real_multiplier(link.force) * length(self.xs[link.i0], self.xs[link.i0 + self.y_offset], self.xs[link.i1], self.xs[link.i1 + self.y_offset]) * self.COST_PER_M
        return cost, real_cost
    
    # def real_cost(self, xs):
    #     cost = len(self.joints) * self.COST_PER_JOINT

    #     for link in self.links:
    #         force = float(Function('force', [self.xs], [link.force])(xs))
    #         length = float(Function('length', [self.xs], [link.length])(xs))
    #         multiplier = 1
    #         if(force < 0):
    #             multiplier = int(-force/6)+1
    #         else:
    #             multiplier = int(force/9)+1
    #         cost += length*self.COST_PER_M*multiplier
    #     return cost

    def compute_jacobian(self):
        return jacobian(self.cost, self.xs)
    
    def compute_hessian(self):
        [H, g] = hessian(self.cost, self.xs)
        return H

    def print_joint_forces(self):
        for i, joint in enumerate(self.joints):
            print("Joint", i, joint.fx, joint.fy)

    def print_link_forces(self):
        for i, link in enumerate(self.links):
            print("Link", i, link.force)

    def plot(self, xs, title = "", plot_external = False, plot_member = False):

            for link in self.links:
                plt.plot([xs[link.i0], xs[link.i1]], [xs[link.i0+self.y_offset], xs[link.i1+self.y_offset]], color='black', marker='o', linewidth=1)
                plt.annotate('{0}'.format(link.i0), xy=(xs[link.i0], xs[link.i0+self.y_offset]), xytext=(2, 2), textcoords='offset points')
                plt.annotate('{0}'.format(link.i1), xy=(xs[link.i1], xs[link.i1+self.y_offset]), xytext=(2, 2), textcoords='offset points')

            for i, joint in enumerate(self.joints):
                x = xs[i]
                y = xs[i+self.y_offset]
                fx = float(Function('fx', [self.xs], [joint.fx])(xs))
                fy = float(Function('fy', [self.xs], [joint.fy])(xs))
                fex = float(Function('fex', [self.xs], [joint.fex])(xs))
                fey = -float(Function('fey', [self.xs], [joint.fey])(xs))
                # print(i, fex, fey)
                if((joint.fex != 0 or fey != 0)) and plot_external:
                    mag = round(math.sqrt(fex*fex+fey*fey), 3)
                    plt.arrow(xs[i], xs[i+self.y_offset], fex/10, fey/10, width=.2, label="force")
                    # plt.annotate('{0}kN'.format(str(mag)), xy=(joint.x+joint.fex, joint.y+joint.fey), xytext=(0, 0), textcoords='offset points')
                    t = plt.text(xs[i]+fex/10, xs[i+self.y_offset]+fey/10, '{0}kN'.format(mag), fontsize=5)
                    t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='grey'))
                if((fx != 0 or fy != 0)) and plot_member:
                    mag = round(math.sqrt(fx*fx+fy*fy), 3)
                    plt.arrow(xs[i], xs[i+self.y_offset], fx/10, fy/10, width=.2, label="force")
                    t = plt.text(xs[i]+fx/10, xs[i+self.y_offset]+fy/10, '{0}kN'.format(mag), fontsize=5)
                    t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='grey'))
            
            if plot_member:
                for link in self.links:
                    x0 = xs[link.i0]
                    y0 = xs[link.i0+self.y_offset]
                    x1 = xs[link.i1]
                    y1 = xs[link.i1+self.y_offset]

                    mx = (x1+x0)/2
                    my = (y1+y0)/2

                    force = round(float(Function('force', [self.xs], [link.force])(xs)), 3)
                    color = 'blue' if force > 0 else 'red'
                    t = plt.text(mx, my, '{0}kN'.format(force), fontsize=5)
                    t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))

            plt.title('Cost: ${:.2f}'.format(float(Function("cost_fn_title", [self.xs], [self.real_cost])(xs))))
            plt.show()
