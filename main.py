# Import the game environment
import GameEnv
# Import pygame for game handling
import pygame
# Import numpy for numerical operations
import numpy as np
# Import DDQNAgent class for deep Q-learning
from ddqn_keras import DDQNAgent
# Import deque for memory buffer
from collections import deque
# Import random and math for utility functions
import random, math

# Define maximum game time for one episode
TOTAL_GAMETIME = 1000
# Define total number of episodes for training
N_EPISODES = 10000
# Define frequency for updating target network
REPLACE_TARGET = 50

# Initialize the racing game environment
game = GameEnv.RacingEnv()
# Set game frame rate to 60 FPS
game.fps = 60

# Initialize game time counter
GameTime = 0
# Initialize list to store game history
GameHistory = []
# Initialize flag to control rendering
renderFlag = False

# Initialize DDQN agent with specified parameters
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, 
                       epsilon_end=0.10, epsilon_dec=0.9995, replace_target=REPLACE_TARGET, 
                       batch_size=512, input_dims=19)

# Commented-out code to load an existing model
# If uncommented, loads a pre-trained model (careful, may overwrite existing model)
# ddqn_agent.load_model()

# Initialize lists to store scores and epsilon history
ddqn_scores = []
eps_history = []

# Main function to run the training loop
def run():
    # Loop through episodes
    for e in range(N_EPISODES):
        # Reset the game environment for a new episode
        game.reset()
        
        # Initialize done flag
        done = False
        # Initialize score for the episode
        score = 0
        # Initialize counter for no-reward ticks
        counter = 0
        
        # Take initial action (no-op) and get initial observation, reward, and done status
        observation_, reward, done = game.step(0)
        # Convert observation to numpy array
        observation = np.array(observation_)

        # Reset game time for the episode
        gtime = 0
        
        # Set render flag (default False, can be set True for all episodes)
        renderFlag = False
        
        # Enable rendering every 10 episodes
        if e % 10 == 0 and e > 0:
            renderFlag = True

        # Main episode loop
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                # Exit if user closes window
                if event.type == pygame.QUIT:
                    return

            # Choose action based on current observation
            action = ddqn_agent.choose_action(observation)
            # Perform action and get new state, reward, and done status
            observation_, reward, done = game.step(action)
            # Convert new observation to numpy array
            observation_ = np.array(observation_)

            # Increment counter if no reward is received
            if reward == 0:
                counter += 1
                # End episode if no reward for 100 ticks
                if counter > 100:
                    done = True
            else:
                # Reset counter if reward is received
                counter = 0

            # Accumulate reward
            score += reward

            # Store experience in agent's memory
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            # Update observation for next iteration
            observation = observation_
            # Train the agent
            ddqn_agent.learn()
            
            # Increment game time
            gtime += 1

            # End episode if max game time reached
            if gtime >= TOTAL_GAMETIME:
                done = True

            # Render game if renderFlag is True
            if renderFlag:
                game.render(action)

        # Store epsilon value for this episode
        eps_history.append(ddqn_agent.epsilon)
        # Store score for this episode
        ddqn_scores.append(score)
        # Calculate average score over last 100 episodes
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        # Update target network every REPLACE_TARGET episodes
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        # Save model every 10 episodes after the first 10
        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
        # Print episode statistics
        print('episode: ', e, 'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)

# Run the training loop
run()