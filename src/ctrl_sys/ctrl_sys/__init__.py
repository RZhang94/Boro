import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Any
from matplotlib import pyplot as plt


class CtrlCallable:
    func: Callable
    size: int
    def __init__(self)-> None:
        pass
    
    def __call__(self, states: dict) -> Any:
        return self.func(kwargs=states)

class Results:
    def __init__(self, 
                time: np.ndarray, 
                x: np.ndarray,
                r: np.ndarray,
                u: np.ndarray, 
                y: np.ndarray,
                **kwargs):
        """
        Stores system data from simulations

        Args:
            time (np.ndarray): time of the system
            x (np.ndarray): states of the system at the corresponding time index
            u (np.ndarray): effect of inputs on the states (Bu) at the corresponding time index
            y (np.ndarray): output states at the corresponding time index
            **kwargs: x_labels, u_labels, y_labels
        """
        self.time = time
        self.x = x
        self.r = r
        self.u = u
        self.y = y

        self.check_labels(**kwargs)

    def plot(self, save_location: Optional[str] = None):
        """
        Plots the system over time. Saves the plot if given target location or shows the plot.
        
        Args:
            save_location (Optional[str], optional): Location to save the plot. Defaults to None.
        """
        self.fig, ax = plt.subplots(3, 1)
        self.fig.set_figheight(9)
        self.fig.set_figwidth(14)

        if len(self.x.shape) == 2:
            for i in range(self.x.shape[0]):
                ax[0].plot(self.time, self.x[i], label=self.x_labels[i])
                ax[0].plot(self.time, self.r[i], label=self.x_labels[i] + ' Reference', linestyle='--')
        else:
            ax[0].plot(self.time, self.x, label=self.x_labels)
            ax[0].plot(self.time, self.r, label=self.x_labels + ' Reference', linestyle='--')
        ax[0].legend()
        ax[0].set_title('States')

        if len(self.u.shape) == 2:
            for i in range(self.u.shape[0]):
                ax[1].plot(self.time, self.u[i], label=self.u_labels[i])
        else:
            ax[1].plot(self.time, self.u, label=self.u_labels)
        ax[1].legend()
        ax[1].set_title('Inputs')

        if len(self.y.shape) == 2:
            for i in range(self.y.shape[0]):
                ax[2].plot(self.time, self.y[i], label=self.y_labels[i])
        else:
            ax[2].plot(self.time, self.y, label=self.y_labels)
        ax[2].legend()
        ax[2].set_title('Outputs')

        if save_location is not None:
            plt.savefig(save_location)
        else:
            plt.show()

    def check_labels(self, **kwargs):
        """
        Checks if the labels are given. If not, assigns default labels.
        
        Args:
            **kwargs: x_labels, u_labels, y_labels <- All are lists
        """
        self.x_labels = kwargs.get('x_labels', None)
        self.u_labels = kwargs.get('u_labels', None)
        self.y_labels = kwargs.get('y_labels', None)

        if self.x_labels is None:
            self.x_labels = ['x' + str(i) for i in range(self.x.shape[0])]
        if self.u_labels is None:
            self.u_labels = ['u' + str(i) for i in range(self.u.shape[0])]
        if self.y_labels is None:
            self.y_labels = ['y' + str(i) for i in range(self.y.shape[0])]

class System:
    def __init__(self,
                total_time: float, 
                time_step: float,
                func_x: CtrlCallable, 
                x0: Union[np.ndarray],
                ref_x: Optional[CtrlCallable] = None,
                func_u: Optional[CtrlCallable] = None, 
                func_y: Optional[CtrlCallable] = None,
                func_r: Union[None, CtrlCallable] = None,
                **kwargs
                ) -> None:
        """
        Simulates a dynamic system with given functions and initial conditions.
        Can be for DT/CT systems in the structure dx = f(x, u), u = g(x, r), y = h(x, u), r = r(t)\
        All outputs should be np.ndarrays of shape (n,).
        """
        self.total_time = total_time
        self.time_step = time_step
        self.func_x = func_x
        self.x0 = x0
        self.func_u = func_u
        self.func_y = func_y
        self.func_r = func_r
        self.ref_x = ref_x

        self.test_structure()

        _time = np.arange(0, total_time, time_step)
        _x = np.zeros((self.func_x.size, len(_time)))
        _u = np.zeros((self.func_u.size, len(_time)))
        _y = np.zeros((self.func_y.size, len(_time)))
        _ref = np.zeros((self.func_r.size, len(_time)))

        res_kwargs = self.handle_labels(**kwargs)
        
        self.results = Results(_time, _x, _ref, _u, _y, **res_kwargs)

    def test_structure(self) -> None:
        x_size = self.func_x.size
        if self.func_r is None:
            self.func_r = ZeroFunc
        r = self.func_r({'x': self.x0, 't': 0})
        
        if self.func_u is None:
            self.func_u = ZeroFunc
        u = self.func_u({'x': self.x0, 'r': r, 't': 0})

        if self.func_y is None:
            self.func_y = ZeroFunc
        y = self.func_y({'x': self.x0, 'u': u, 't': 0})

        x = self.func_x({'x': self.x0, 'u': u, 't': 0})

        valid = True
        if r.shape[0] != x.shape[0]:
            print('r and x are not compatible.')
            valid = False
        if u.shape[0] != x.shape[0]:
            print('u and x are not compatible.')
            valid = False
        if x.shape[0] != x_size:
            print('x and x0 are not compatible.')
            valid = False

        if not valid:
            raise ValueError('System structure is not valid.')
        
        return valid

        



    def handle_labels(self, **kwargs) -> dict:
        res_kwargs = {}
        if 'x_labels' in kwargs:
            res_kwargs['x_labels'] = kwargs['x_labels']
        if 'u_labels' in kwargs:
            res_kwargs['u_labels'] = kwargs['u_labels']
        if 'y_labels' in kwargs:
            res_kwargs['y_labels'] = kwargs['y_labels']
        return res_kwargs

    def simulate(self):
        pass

class ZeroFunc(CtrlCallable):
    def __init__(self):
        self.size = -1

    def __call__(self, **kwargs) -> np.ndarray:
        self.size = kwargs['x'].shape[0]
        return np.zeros(self.size)
