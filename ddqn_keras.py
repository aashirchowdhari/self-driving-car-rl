from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

# ReplayBuffer is used to store and sample experiences (transitions) for training
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size  # Maximum number of transitions to store
        self.mem_cntr = 0  # Counter to keep track of how many transitions have been stored
        self.discrete = discrete  # Boolean flag indicating if the action space is discrete
        # Memory arrays to store state, new state, actions, rewards, and terminal flags
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32  # Choose data type based on discrete actions
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    # Store a transition into the replay buffer
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  # Circular buffer logic
        self.state_memory[index] = state  # Store current state
        self.new_state_memory[index] = state_  # Store next state

        # One-hot encode actions if action space is discrete
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.reward_memory[index] = reward  # Store reward
        self.terminal_memory[index] = 1 - done  # Store done flag (1 for not done, 0 for done)
        self.mem_cntr += 1  # Increment memory counter

    # Sample a batch of transitions from the memory for training
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # Restrict to filled part of buffer
        batch = np.random.choice(max_mem, batch_size)  # Randomly select batch indices

        # Fetch the corresponding samples
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# DDQNAgent handles action selection, learning and target model update
class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995,  epsilon_end=0.10,
                 mem_size=25000, fname='ddqn_model.h5', replace_target=25):
        # Initialize hyperparameters
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_dec = epsilon_dec  # Epsilon decay rate
        self.epsilon_min = epsilon_end  # Minimum epsilon
        self.batch_size = batch_size
        self.model_file = fname  # File to save/load model
        self.replace_target = replace_target  # Interval for updating target network
        # Initialize replay memory
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        # Initialize evaluation and target networks
        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

    # Store transition in replay buffer
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # Epsilon-greedy action selection
    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]  # Reshape for model input

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)  # Random action (exploration)
        else:
            actions = self.brain_eval.predict(state)  # Predict Q-values
            action = np.argmax(actions)  # Choose action with highest Q-value

        return action

    # Train the evaluation network using a batch of experiences
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            # Sample a batch
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            # Convert one-hot action to index
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            # Predict Q-values
            q_next = self.brain_target.predict(new_state)  # Q-values from target network
            q_eval = self.brain_eval.predict(new_state)  # Q-values from eval network (for Double DQN)
            q_pred = self.brain_eval.predict(state)  # Q-values for current state

            max_actions = np.argmax(q_eval, axis=1)  # Select max Q actions using eval network

            q_target = q_pred  # Initialize targets with current predictions

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # Update only the selected action values
            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.brain_eval.train(state, q_target)  # Train eval network on updated targets

            # Decay epsilon
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    # Copy weights from eval network to target network
    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    # Save the eval model to disk
    def save_model(self):
        self.brain_eval.model.save(self.model_file)

    # Load a model from disk and update both networks
    def load_model(self):
        self.brain_eval.model = load_model(self.model_file)
        self.brain_target.model = load_model(self.model_file)

        # Ensure target model is updated only if needed
        if self.epsilon == 0.0:
            self.update_network_parameters()

class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size = 256):
        self.NbrStates = NbrStates               # Number of input features or state dimensions
        self.NbrActions = NbrActions             # Number of possible actions the agent can take
        self.batch_size = batch_size             # Batch size used during training
        self.model = self.createModel()          # Create and initialize the model
    
    def createModel(self):
        # This method builds a neural network using Keras Sequential API
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # First dense layer with 256 neurons and ReLU activation
        model.add(tf.keras.layers.Dense(self.NbrActions, activation=tf.nn.softmax))  # Output layer with softmax to give probability distribution over actions
        model.compile(loss = "mse", optimizer="adam")  # Compile model with Mean Squared Error loss and Adam optimizer

        return model  # Return the compiled model
    
    def train(self, x, y, epoch = 1, verbose = 0):
        # Train the model using provided input (x) and target output (y)
        # Uses the specified batch size and verbosity level
        self.model.fit(x, y, batch_size = self.batch_size , verbose = verbose)

    def predict(self, s):
        # Predict the Q-values (action values) for a batch of states 's'
        return self.model.predict(s)

    def predictOne(self, s):
        # Predict the Q-values for a single state 's'
        # Reshape the input to match model input requirements and flatten the output
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()
    
    def copy_weights(self, TrainNet):
        # Copy weights from another model (TrainNet) to this model
        # Typically used to update the target network with the evaluation network’s weights
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())  # Assign values from TrainNet to this model
