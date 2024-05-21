import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Any
from matplotlib import pyplot as plt


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
            x (np.ndarray): states of the system
            u (np.ndarray): effect of inputs on the states (Bu)
            y (np.ndarray): output states
        """
        self.time = time
        self.x = x
        self.r = r
        self.u = u
        self.y = y

        self.check_labels(**kwargs)

    def plot(self):
        self.fig, ax = plt.subplots(3, 1)
        #fig size in pixels 900x900
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

        plt.show()
        print('hi')



    def check_labels(self, **kwargs):
        self.x_labels = kwargs.get('x_labels', None)
        self.u_labels = kwargs.get('u_labels', None)
        self.y_labels = kwargs.get('y_labels', None)

        if self.x_labels is None:
            self.x_labels = ['x' + str(i) for i in range(self.x.shape[0])]
        if self.u_labels is None:
            self.u_labels = ['u' + str(i) for i in range(self.u.shape[0])]
        if self.y_labels is None:
            self.y_labels = ['y' + str(i) for i in range(self.y.shape[0])]

    def save(self):
        pass


class System:
    def __init__(self,
                total_time: float, 
                time_step: float,
                func_x: Callable, 
                x0: Union[float, np.ndarray],
                ref_x: Optional[Callable] = None,
                func_u: Optional[Callable] = None, 
                func_y: Optional[Callable] = None,
                **kwargs
                ) -> None:
        self.total_time = total_time
        self.time_step = time_step
        self.func_x = func_x
        self.x0 = x0
        self.func_u = func_u
        self.func_y = func_y
        self.ref_x = ref_x

        x_size, y_size = self.verify_size()

        _time = np.arange(0, total_time, time_step)
        _x = np.zeros((x_size, len(_time)))
        _u = np.zeros((x_size, len(_time)))
        _y = np.zeros((y_size, len(_time)))
        
        if self.ref_x is not None:
            _ref = self.ref_x(_time)
        else:
            _ref = np.zeros((x_size, len(_time)))

        res_kwargs = self.handle_labels(**kwargs)
        
        self.results = Results(_time, _x, _ref, _u, _y, **res_kwargs)
        

    def verify_size(self):
        test_val = self.func_x(self.x0)
        x_size = check_size(test_val)

        if self.func_u is not None:
            test_val = self.func_u(self.x0)
            u_size = check_size(test_val)
            if x_size != u_size:
                raise ValueError('The size of the state and input must be the same')
            
        if self.func_y is not None:
            test_val = self.func_y(self.x0)
            y_size = check_size(test_val)
        else:
            y_size = 1

        return x_size, y_size
    
    def form_ref(self, ref_x: Union[float, np.ndarray], time):
        if ref_x is None:
            return np.zeros((1, len(time)))
        elif isinstance(ref_x, float):
            return np.ones((1, len(time))) * ref_x
        elif isinstance(ref_x, np.ndarray):
            if ref_x.shape[1] != len(time):
                raise ValueError('The reference and time must be the same length')
            return ref_x


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

def check_size(value: Any):
    if isinstance(value, np.ndarray):
        return np.shape(value)
    elif isinstance(value, list):
        return len(value)
    elif isinstance(value, float):
        return 1
    elif isinstance(value, int):
        return 1
    else:
        raise ValueError('The value is not a valid type')

test = System(10, 0.1, lambda x: x, 1)
test.results.plot()

print('hi')