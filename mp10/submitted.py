'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    # raise RuntimeError("You need to write this part!")
    M=model.M
    N=model.N
    W=model.W   # wall
    T=model.T   # terminal
    D=model.D
    P=np.zeros((M,N,4,M,N))
    for r in range(M):
        for c in range(N):
            if T[r,c]==1:
                P[r,c,:,:,:]=0
                continue

            for a in range(4):
                if a==0:    # left
                    if c==0:    # left wall
                        if r==0:   # top left corner
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,1] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # down
                            P[r,c,a,r,c]+=D[r,c,2] # up
                            P[r,c,a,r,c]+=D[r,c,0] # left
                        elif r==M-1:   # bottom left corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,2] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # up
                            P[r,c,a,r,c]+=D[r,c,1] # down
                            P[r,c,a,r,c]+=D[r,c,0] # left
                        else:  # left wall, not corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,2] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # up
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,1] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # down
                            P[r,c,a,r,c]+=D[r,c,0] # left
                    else:   # no left wall
                        if r==0:  # top row
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,1] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # down
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,0] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,0] # left
                            P[r,c,a,r,c]+=D[r,c,2] # up
                        elif r==M-1:  # bottom row
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,2] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # up
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,0] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,0] # left
                            P[r,c,a,r,c]+=D[r,c,1] # down
                        else:  # not corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,2] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # up
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,1] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # down
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,0] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # left
                elif a==1:  # up
                    if r==0:   # top wall
                        if c==0:    # top left corner
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,2] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # right
                            P[r,c,a,r,c]+=D[r,c,1] # left
                            P[r,c,a,r,c]+=D[r,c,0] # up
                        elif c==N-1:    # top right corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,1] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # left
                            P[r,c,a,r,c]+=D[r,c,2] # right
                            P[r,c,a,r,c]+=D[r,c,0] # up
                        else:   # top wall, not corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,1] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,1] # left
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,2] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]
                            P[r,c,a,r,c]+=D[r,c,0] # up
                    else:   # no top wall
                        if c==0:    # left column
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,2] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # right
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,0] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # up
                            P[r,c,a,r,c]+=D[r,c,1] # left
                        elif c==N-1:    # right column
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,1] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # left
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,0] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # up
                            P[r,c,a,r,c]+=D[r,c,2] # right
                        else:   # not corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,1] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # left
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,2] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # right
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,0] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # up
                elif a==2:  # right
                    if c==N-1:    # right wall
                        if r==0:    # top right corner
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,2] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,2] # down
                            P[r,c,a,r,c]+=D[r,c,1] # up
                            P[r,c,a,r,c]+=D[r,c,0] # right
                        elif r==M-1:    # bottom right corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,1] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # up
                            P[r,c,a,r,c]+=D[r,c,2] # down
                            P[r,c,a,r,c]+=D[r,c,0] # right
                        else:   # right wall, not corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,1] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # up
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,2] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # down
                            P[r,c,a,r,c]+=D[r,c,0] # right
                    else:   # no right wall
                        if r==0:    # top row
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,2] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # down
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,0] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # right
                            P[r,c,a,r,c]+=D[r,c,1] # left
                        elif r==M-1:    # bottom row
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,1] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # up
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,0] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # right
                            P[r,c,a,r,c]+=D[r,c,2] # down
                        else:   # not corner
                            if W[r-1,c]==0:
                                P[r,c,a,r-1,c]+=D[r,c,1] # up
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # up
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,2] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # down
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,0] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # right
                elif a==3:  # down
                    if r==M-1:    # bottom wall
                        if c==0:    # bottom left corner
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,1] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # right
                            P[r,c,a,r,c]+=D[r,c,2] # left
                            P[r,c,a,r,c]+=D[r,c,0] # down
                        elif c==N-1:    # bottom right corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,2] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # left
                            P[r,c,a,r,c]+=D[r,c,1] # right
                            P[r,c,a,r,c]+=D[r,c,0] # down
                        else:   # bottom wall, not corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,2] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # left
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,1] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # right
                            P[r,c,a,r,c]+=D[r,c,0] # down
                    else:   # no bottom wall
                        if c==0:    # left column
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,1] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # right
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,0] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # down
                            P[r,c,a,r,c]+=D[r,c,2] # left
                        elif c==N-1:    # right column
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,2] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # left
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,0] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # down
                            P[r,c,a,r,c]+=D[r,c,1] # right
                        else:   # not corner
                            if W[r,c-1]==0:
                                P[r,c,a,r,c-1]+=D[r,c,2] # left
                            else:
                                P[r,c,a,r,c]+=D[r,c,2]  # left
                            if W[r,c+1]==0:
                                P[r,c,a,r,c+1]+=D[r,c,1] # right
                            else:
                                P[r,c,a,r,c]+=D[r,c,1]  # right
                            if W[r+1,c]==0:
                                P[r,c,a,r+1,c]+=D[r,c,0] # down
                            else:
                                P[r,c,a,r,c]+=D[r,c,0]  # down
    return P





def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            if model.T[r,c]==1:
                U_next[r,c] = model.R[r,c]
            else:
                U_next[r,c] = model.R[r,c] + model.gamma*np.max(np.sum(P[r,c,:,:,:]*U_current, axis=(1,2)))
    return U_next

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")
    P = compute_transition_matrix(model)
    U = np.zeros((model.M, model.N))
    while True:
        U_next = update_utility(model, P, U)
        if np.max(np.abs(U_next-U)) < epsilon:
            break
        else:
            U = U_next
    return U


if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
