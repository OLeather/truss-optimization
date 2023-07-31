import torch
from sortedcollections import OrderedSet

class Truss():
    FORCE_PER_M = -2.5
    COST_PER_JOINT = 5
    COST_PER_M = 15
    MAX_TENSION = 9 # +
    MAX_COMPRESSION = 6 # -

    def __init__(self, joints, links, bridge_joints, DEVICE):
        '''
        joints: list of integers with size [n_joints] where 1 = gusset, 2 = pin, 3 = roller
        links: list of integers with size [2] where [0] is index of first joint and [1] is index of second joint
        bridge_joints: list of integers with the index of each bridge joint
        '''
        self.joints = joints
        self.n_joints = len(joints)
        self.links = links
        self.n_links = len(links)
        self.bridge_joints = bridge_joints
        self.bridge_links = []
        for i, link in enumerate(links):
            j0 = link[0]
            j1 = link[1]
            if(j0 in bridge_joints and j1 in bridge_joints):
                self.bridge_links.append(i)
        self.joint_connections = {}
        for l_i, l in enumerate(links):
            j0 = l[0]
            j1 = l[1]
            if not j0 in self.joint_connections:
                self.joint_connections[j0] = []
            if not j1 in self.joint_connections:
                self.joint_connections[j1] = []
            self.joint_connections[j0].append((j1, l_i))
            self.joint_connections[j1].append((j0, l_i))
            
        print(self.joint_connections)
        self.DEVICE = DEVICE

    def length(self, p0, p1):
        # print(p0.shape, p1-p0)
        # print(torch.norm(p1-p0, dim=1))
        return torch.norm(p1-p0, dim=1)

    def solve(self, X, print_solution = False):
        '''
        X: tensor of floats with size [batch, n_joints, 2] where [:, joint] = [x,y]

        Goal: construct an equation in the form of A*X = B where X.shape=[batch, n_links, 2] contains each link force
        '''
        batch = X.size(dim=0)

        lengths = torch.stack([self.length(X[:,l[0]], X[:,l[1]]) for l in self.links]).T # shape=[batch, n_links]
        link_forces = [lengths[:,i]*self.FORCE_PER_M/2.0 for i in self.bridge_links] # shape=[batch, n_bridge_links]
        external_forces = torch.zeros(self.n_joints, batch, 2).to(self.DEVICE) # shape=[batch, n_joints, 2] where [:, n] = [fx, fy]
        for i, l in enumerate(self.bridge_links):
            j0 = self.links[l][0]
            j1 = self.links[l][1]
            external_forces[j0, :] += torch.hstack([torch.zeros(batch, 1).to(self.DEVICE), link_forces[i][None, :].T])
            external_forces[j1, :] += torch.hstack([torch.zeros(batch, 1).to(self.DEVICE), link_forces[i][None, :].T])
        

        external_forces = torch.transpose(external_forces, 0, 1).flatten(1) # shape = [batch, n_joints*2]
        unknowns = OrderedSet() # contains the key of each unknown variable formatted as F_{link} or N_{joint}
        equations = []
        for j0 in range(self.n_joints):
            connected_joints = self.joint_connections[j0] # array of tuples of type (connected joint index, link index)
            fnetX = {}
            fnetY = {}
            for j1, l_i in connected_joints:
                member_force = 'F_{0}'.format(l_i)
                unknowns.add(member_force)
                # U = (X1-X0)/||X1-X0||
                # print(X[:, j1])
                # print(self.length(X[:,j0], X[:,j1]))
                U = (X[:,j1]-X[:,j0])/self.length(X[:,j0], X[:,j1])[:,None]
                fnetX[member_force] = U[:, 0]
                fnetY[member_force] = U[:, 1]

                # if ROLLER or PIN joint, add a normal force in y axis
                if self.joints[j0] == 2 or self.joints[j0] == 3:
                    normal_force = 'N_y_{0}'.format(j0)
                    unknowns.add(normal_force)
                    fnetY[normal_force] = torch.ones(batch).to(self.DEVICE)
                # if PIN joint, add a normal force in x axis
                if self.joints[j0] == 2:
                    normal_force = 'N_x_{0}'.format(j0)
                    unknowns.add(normal_force)
                    fnetX[normal_force] = torch.ones(batch).to(self.DEVICE)
            
            if(print_solution):
                print('fnetX_{0} = {1}'.format(j0, fnetX))
                print('fnetY_{0} = {1}'.format(j0, fnetY))
            equations.append(fnetX)
            equations.append(fnetY)
        
        A = []
        B = []
        member_indices = []
        for i, unknown in enumerate(unknowns):
            if(unknown.split('_')[0] == 'F'):
                member_indices.append(i)
        for i, equation in enumerate(equations):
            row = []
            for unknown in unknowns:
                if unknown in equation:
                    row.append(equation[unknown][:,None])
                else:
                    row.append(torch.zeros(batch, 1).to(self.DEVICE))
            A.append(torch.hstack(row)[:, None, :])
            B.append(-external_forces[:,i])

        A = torch.hstack(A)
        B = torch.stack(B).T

        X = torch.linalg.solve(A, B)
        
        if(print_solution):
            for i, unknown in enumerate(unknowns):
                print(unknown, '=', X[:, i])

        relu = torch.nn.ReLU()

        multipliers = torch.round((relu(X)/9.0 + relu(-X)/9.0) + 1)[:, member_indices]
        cost = torch.tensor([self.COST_PER_JOINT*self.n_joints]*batch, dtype=torch.float32).to(self.DEVICE)
        cost += torch.sum(lengths*multipliers, dim=1)*self.COST_PER_M

        return X, cost



        

    