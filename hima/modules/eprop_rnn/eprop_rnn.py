import numpy as np

def softmax(x: np.ndarray, temp=1.) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)

class RNNWithEProp:

    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float
    decay_rate: float

    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 learning_rate: float = 0.01, 
                 decay_rate: float = 0.9,
                 seed: int = 42):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.decay = decay_rate
        
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        self.e_xh = np.zeros_like(self.W_xh)
        self.e_hh = np.zeros_like(self.W_hh)
        self.e_hy = np.zeros_like(self.W_hy)
        
        self.hidden_states = []
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        if not self.hidden_states:
            h_prev = np.zeros((1, self.hidden_size))
        else:
            h_prev = self.hidden_states[-1]
        
        h = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        y = np.dot(h, self.W_hy) + self.b_y

        self.hidden_states.append(h)
        return h, softmax(y)
    
    def decode(self, hidden):
        return softmax(np.dot(hidden, self.W_hy) + self.b_y).squeeze()

    def compute_learning_signals(self, y, target):

        error = y - target
        L_y = error 
        
        L_h = np.dot(L_y, self.W_hy.T) * (1 - self.hidden_states[-1]**2)
        
        return L_y, L_h
    
    def update_eligibility_traces(self, x, L_h):

        self.e_xh = self.decay * self.e_xh + np.outer(x, L_h)
        self.e_hh = self.decay * self.e_hh + np.outer(self.hidden_states[-2] if len(self.hidden_states) > 1 
                                                     else np.zeros((1, self.hidden_size)), L_h)
    
    def backward(self, x, L_y, L_h):
        dW_hy = np.outer(self.hidden_states[-1], L_y)
        dW_xh = self.e_xh
        dW_hh = self.e_hh
        
        self.W_hy -= self.lr * dW_hy
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh

        self.b_y -= self.lr * L_y
        self.b_h -= self.lr * L_h
    
    def reset_states(self):
        self.hidden_states = []
        self.e_xh = np.zeros_like(self.W_xh)
        self.e_hh = np.zeros_like(self.W_hh)
        self.e_hy = np.zeros_like(self.W_hy)
    
    def train_step(self, x_seq):
        total_loss = 0
        
        for t in range(len(x_seq) - 1):
            x = x_seq[t].reshape(1, -1)
            target = x_seq[t + 1][:self.output_size].reshape(1, -1)
            
            # Forward pass
            _, y = self.forward(x)
            
            loss = np.mean(0.5 * (y - target)**2)
            total_loss += loss

            L_y, L_h = self.compute_learning_signals(y, target)
            
            if t > 0:
                self.update_eligibility_traces(x, L_h)
            
            self.backward(x, L_y, L_h)
        
        return total_loss / len(x_seq)
    
    def predict(self, x_seq):
        predictions = []
        self.reset_states()
        
        for x in x_seq:
            x = x.reshape(1, -1)
            _, y = self.forward(x)
            predictions.append(y.copy())
        
        self.reset_states()
        return np.vstack(predictions)
    
    def predict_one(self, x):
        
        x = x.reshape(1, -1)
        _, y = self.forward(x)

        return y.flatten()
    
    def validate(self, x_val):
        accuracy = 0
        for seq in x_val:
            correct_seq = 0
            total_seq = 0
            self.reset_states()
            for t in range(len(seq) - 1):
                x = seq[t].reshape(1, -1)
                target = seq[t + 1][:self.output_size]
                
                _, y = self.forward(x)
                pred = np.argmax(y, axis=1)
                true = np.argmax(target)
                
                if pred == true:
                    correct_seq += 1
                total_seq += 1
            accuracy += correct_seq / total_seq
            self.reset_states()
        
        return accuracy / x_val.shape[0]