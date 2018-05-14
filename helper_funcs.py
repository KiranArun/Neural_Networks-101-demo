import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_text=None):

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            circle = plt.Circle((x,y), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                text = layer_text[n][m]
                ax.annotate(text, xy=(x, y), zorder=5, ha='center', va='center', fontsize=15)


    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
                
                
                
def loss_optimization(ax):
    
    a = np.linspace(-3, 3, 601)
    b = 2*a**2+3
    dbda = 4*a

    ax[0].plot(a,b, label='loss function')
    ax[0].plot(a,dbda, label='loss derivative')
    
    idx = 100
    
    ax[1].plot(a[idx],b[idx], 'o', label='current loss value')
    ax[1].plot(a[idx],dbda[idx], 'o', label='current loss derivative')
    ax[1].plot(a,dbda[idx]*a+(b[idx]-dbda[idx]*a[idx]), label='current loss tangent')
    
    idx2 = 450
    
    ax[2].plot(a[idx2],b[idx2], 'o', label='current loss value')
    ax[2].plot(a[idx2],dbda[idx2], 'o', label='current loss derivative')
    ax[2].plot(a,dbda[idx2]*a+(b[idx2]-dbda[idx2]*a[idx2]), label='current loss tangent')
    
    for axes in ax:
        axes.axhline(0,color='black',alpha=0.5)
        axes.axvline(0,color='black',alpha=0.5)
        axes.set_xlabel('W (variable)')
        axes.grid(True)
        axes.legend()
        axes.set_xlim(-3, 3)
        axes.set_ylim(min(np.min(b),np.min(dbda)), max(np.max(b),np.max(dbda)))
        
def softmax_graphs(ax):
    x = np.arange(10)
    values = np.array([1,-2,0,2,-1,1,2,-2,6,2], dtype=np.float32)

    ax[0].bar(x, values)
    ax[1].bar(x, (values-np.min(values))/np.sum(values-np.min(values)))
    argmax_vals = np.zeros_like(values)
    argmax_vals[np.argmax(values)] = 1
    ax[2].bar(x, argmax_vals)
    ax[3].bar(x, np.array([np.exp(v) for v in values])/np.sum(np.array([np.exp(v) for v in values])))


    ax[0].set_title('Raw outputs from NN')
    ax[1].set_title('Normalised')
    ax[2].set_title(r'Hardmax ($\arg \max$)')
    ax[3].set_title('Softmax')

    for axes in ax:
        plt.sca(axes)
        plt.xticks(x, [r'$k_1$',r'$k_2$',r'$k_3$', r'$k_4$', r'$k_5$',r'$k_6$', r'$k_7$', r'$k_8$',r'$k_9$', r'$k_{10}$'])
        plt.grid(True)

def animate_gradient_descent(L_func=None, L_func_p=None, frames=20, lr=0.1, X=np.linspace(-10,10,2001), interval=320):

    if L_func or L_func_p == None:
        def L_func(x):
            return(x**2)

        def L_func_p(x):
            return(2*x)

    xs = np.array([-10])
    ys = np.array([L_func(xs[0])])

    for i in range(frames):
        x = xs[-1] - lr*L_func_p(xs[-1])
        y = L_func(x)
        xs = np.append(xs,x)
        ys = np.append(ys,y)

    fig, ax = plt.subplots()

    ax.plot(X,f(X))

    line, = ax.plot([], [], 'o')
      
    def animate(i):
      line.set_data(xs[i], ys[i])
      return (line,)

    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames, interval=interval, 
                                   blit=True)

    return(anim)