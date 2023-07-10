from casadi import *
from truss import JointType
from sortedcollections import OrderedSet

class Joint():
    def __init__(self, type):
        self.type = type
        self.fex = 0
        self.fey = 0
        self.fx = 0
        self.fy = 0
    def set_fx(self, fx):
        self.fx = fx
    
    def set_fy(self, fy):
        self.fy = fy

class Link():
    def __init__(self, i0, i1):
        self.i0 = i0
        self.i1 = i1
        self.force = 0
        self.length = 0
    
    def set_force(self, force):
        self.force = force

class Force():
    def __init__(self, fx, fy, joints):
        self.fx = fx
        self.fy = fy
        self.joints = joints

def length(x0, y0, x1, y1):
    return sqrt(constpow(x0-x1,2) + constpow(y0-y1,2))


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
        self.cost = self.compute_cost_function()
        self.jacobian = self.compute_jacobian()
        self.hessian = self.compute_hessian()

    def apply_link_lengths(self):
        for i, link in enumerate(self.links):
            link.length = length(self.xs[link.i0], self.xs[link.i0 + self.y_offset], self.xs[link.i1], self.xs[link.i1 + self.y_offset])
            link.force = SX.sym('F{0}'.format(i))

    def apply_external_forces(self):
        for force in self.forces:
            if len(force.joints) == 1:
                self.joints[force.links[0]].fex = force.fx
                self.joints[force.links[0]].fey = force.fy
            elif len(force.joints) > 1:
                for i in range(len(force.joints)-1):
                    j = force.joints[i]
                    k = force.joints[i+1]
                    self.joints[i].fex += force.fx * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2
                    self.joints[i].fey += force.fy * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2 
                    self.joints[k].fex += force.fx * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2
                    self.joints[k].fey += force.fy * length(self.xs[j], self.xs[j + self.y_offset], self.xs[k], self.xs[k + self.y_offset]) / 2 

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
            # print(equation_names[i], "=", eqn)
        # print(A)
        As = []
        for row in A:
            As.append(horzcat(*row))
        
        Ab = vertcat(*As)

        # print(Ab)
        solution = solve(Ab, self.external_forces)
        
        # print(solution.shape)
        for i, assign in enumerate(unknown_assign.values()):
            assign(solution[i])
        
        return solution


    def compute_cost_function(self):
        # https://www.desmos.com/calculator/ao7fjllkuj
        def multiplier(x):
            e = 2.71828
            p = 10
            b = 23
            n = 8.885
            m = 17.885
            j = 5.885
            k = 11.885
            return constpow(1/(1+constpow(e,-b*(-x-n))), p) \
                + constpow(1/(1+constpow(e,-b*(-x-m))), p) \
                + constpow(1/(1+constpow(e,-b*(x-j))), p) \
                + constpow(1/(1+constpow(e,-b*(x-k))), p) + 1

        
        cost = len(self.joints) * self.COST_PER_JOINT

        for link in self.links:
            cost += multiplier(link.force) * length(self.xs[link.i0], self.xs[link.i0 + self.y_offset], self.xs[link.i1], self.xs[link.i1 + self.y_offset]) * self.COST_PER_M
        
        return cost

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