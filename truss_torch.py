import torch
import math
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
    
    def set_force(self, f):
        self.force = f

    def set_length(self, l):
        self.length = l

class Force():
    def __init__(self, fx, fy, joints):
        self.fx = fx
        self.fy = fy
        self.joints = joints

def length(x0, y0, x1, y1):
    return torch.sqrt(pow(x0-x1,2) + pow(y0-y1,2))

class Truss():
    FORCE_PER_M = 5 # kg/m
    COST_PER_JOINT = 5 # Dollars
    COST_PER_M = 15 # Dollars

    def __init__(self, joints, links, forces, DEVICE):
        self.joints = joints
        self.links = links
        self.forces = forces
        self.y_offset = len(joints)
        self.DEVICE = DEVICE

    def solve(self, Y):
        Y = Y.T
        # costs = torch.zeros(Y.size(dim=0), 1).to(self.DEVICE)
        self.apply_link_lengths(Y)
        self.apply_external_forces(Y)
        self.construct_knowns_vector()
        costs = self.construct_equations_matrix(Y)
        return costs

    def apply_link_lengths(self, X):
        for i, link in enumerate(self.links):
            l = torch.tensor(length(X[link.i0], X[link.i0 + self.y_offset], X[link.i1], X[link.i1 + self.y_offset]), requires_grad=True).to(self.DEVICE)
            link.set_length(l)
            link.force = 'F{0}'.format(i)


    def apply_external_forces(self, X):
        for joint in self.joints:
            joint.fex = torch.zeros(X.size(dim=1)).to(self.DEVICE)
            joint.fey = torch.zeros(X.size(dim=1)).to(self.DEVICE)
            joint.fx = torch.zeros(X.size(dim=1)).to(self.DEVICE)
            joint.fy = torch.zeros(X.size(dim=1)).to(self.DEVICE)
        for force in self.forces:
            if len(force.joints) == 1:
                self.joints[force.links[0]].fex = torch.tensor(force.fx, requires_grad=True).to(self.DEVICE)
                self.joints[force.links[0]].fey = torch.tensor(force.fy, requires_grad=True).to(self.DEVICE)
            elif len(force.joints) > 1:
                for i in range(len(force.joints)-1):
                    j = force.joints[i]
                    k = force.joints[i+1]
                    self.joints[i].fex += force.fx * length(X[j], X[j + self.y_offset], X[k], X[k + self.y_offset]) / 2
                    self.joints[i].fey += force.fy * length(X[j], X[j + self.y_offset], X[k], X[k + self.y_offset]) / 2 
                    self.joints[k].fex += force.fx * length(X[j], X[j + self.y_offset], X[k], X[k + self.y_offset]) / 2
                    self.joints[k].fey += force.fy * length(X[j], X[j + self.y_offset], X[k], X[k + self.y_offset]) / 2 
                

    def construct_knowns_vector(self):
        knowns = []
        for joint in self.joints:
            knowns.append(joint.fex)
            knowns.append(joint.fey)
        self.external_forces = torch.cat(knowns)

    def construct_equations_matrix(self, X):
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
                xi = X[i]
                xj = X[j]
                yi = X[i + self.y_offset]
                yj = X[j + self.y_offset]
                
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
                force = 'N{0}y'.format(i)
                fnetY[force] = torch.ones(X.size(dim=1)).to(self.DEVICE)
                forces.add(force)
                unknown_assign[force] = joint.set_fy
                # if force not in forces:
                #     forces.append(force)

            if joint.type == JointType.PIN:
                force = 'N{0}x'.format(i)
                fnetX[force] = torch.ones(X.size(dim=1)).to(self.DEVICE)
                forces.add(force)
                unknown_assign[force] = joint.set_fx
                # if force not in forces:
                #     forces.append(force)
            
            equations.append(fnetX)
            equation_names.append('Fnetx{0}'.format(i))
            equations.append(fnetY)
            equation_names.append('Fnety{0}'.format(i))

        A, B = self.construct_system(equations, self.external_forces, forces, X)
        # print(A.shape)
        X = (torch.linalg.inv(A.T) @ B).T

        for i, force in enumerate(forces):
            unknown_assign[force](X[i])
            # print(force, "=", X[i])
        
        cost = self.compute_cost_function(X)
        return cost

    def construct_system(self, equations, knowns, unknowns, X):
        A = []
        B = []
        
        for i, equation in enumerate(equations):
            row = []
            for unknown in unknowns:
                if unknown in equation:
                    row.append(equation[unknown])
                else:
                    row.append(torch.zeros(X.size(dim=1)).to(self.DEVICE))
            A.append(torch.stack(row))
            # print(knowns[i])
            B.append(knowns[i])

        # print(A)
        # print(B)
        # return torch.cat(A), torch.cat(B)
        return torch.stack(A), torch.stack(B)
        # return torch.tensor(A, requires_grad=True).to(self.DEVICE), torch.tensor(B, requires_grad=True).to(self.DEVICE)
    
    def compute_cost_function(self, X):
        # https://www.desmos.com/calculator/ao7fjllkuj
        def multiplier(x):
            e = 2.71828
            p = 10
            b = 23
            n = 8.885
            m = 17.885
            j = 5.885
            k = 11.885
            return pow(1/(1+pow(e,-b*(-x-n))), p) \
                + pow(1/(1+pow(e,-b*(-x-m))), p) \
                + pow(1/(1+pow(e,-b*(x-j))), p) \
                + pow(1/(1+pow(e,-b*(x-k))), p) + 1

        
        cost = torch.tensor([float(len(self.joints))]*X.size(dim=1), requires_grad=True).to(self.DEVICE) * self.COST_PER_JOINT

        for link in self.links:
            cost += multiplier(link.force) * link.length * self.COST_PER_M
        
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