import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def update(i, fig, ax):
    """
    Update angle to create the animation effect on the gif plot
    """
    ax.view_init(elev=20., azim=i)
    return fig, ax

def plot_3d_surface(Xs, Ys, Zs, save_gif = False):
    """
    Plots a 3d surface from non-linear 3d-points 
    :param Xs: list with X Coords
    :param Ys: list with Y Coords
    :param Zs: list with Z Coords
    """    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    
    fig.tight_layout()
    
    plt.xlabel('rho')
    plt.ylabel('alpha')
    plt.title('map score') 
    fig.savefig('3D.png')
    
    if save_gif:
        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), repeat=True, fargs=(fig, ax)) 
        anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
    

def grid_search(min_alpha = 0, max_alpha = 10 , min_rho = 0, max_rho = 1, 
                max_division = 10, max_iterations = 1, plot_results = True, save_gif = False):
    """
    Computes a grid shearh comvining different rho and alpha values 
    :param max_division: int, indicates the division level for the grid
    :param max_iterations: int, Fixes the level of recursivity of the search 
    :param plot_results: bool, enables 3d plot from results
    :return: List with the GridShear Results (rho_plt, alpha_plt, result_plt)
    """      
    best_result, best_rho, best_alpha = 0, 0, 0
    rho_plt, alpha_plt, result_plt = [], [], []
 
    
    for iteration in range(max_iterations):
        for alpha in np.linspace(min_alpha, max_alpha, max_division):
            for rho in np.linspace(min_rho, max_rho, max_division):
                #Compute here
                new_result = random()
                #print("Result for rho:",rho," and alpha:",alpha," is:",new_result)
                rho_plt.append(rho)
                alpha_plt.append(alpha)
                result_plt.append(new_result)
                
                if new_result > best_result:
                    best_result = new_result
                    best_rho = rho
                    best_alpha = alpha
                    
        # update new limits for next iteration            
        min_alpha = best_alpha - (max_alpha/max_division)/2
        max_alpha = best_alpha + (max_alpha/max_division)/2        
        min_rho = best_rho - (max_rho/max_division)/2
        max_rho = best_rho + (max_rho/max_division)/2
        
    if plot_results:
        plot_3d_surface(rho_plt, alpha_plt, result_plt, save_gif = save_gif)
        
    return best_alpha, best_rho
        
        
if __name__ == "__main__":
    grid_search()
    

