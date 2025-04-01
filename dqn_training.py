import numpy as np
import time
import os
import gc
import matplotlib.pyplot as plt
import torch
from alternative_env import create_mario_env
from standard_dqn import DQNAgent
from prioritized_replay import PrioritizedReplayBuffer
import argparse

os.makedirs('graphs', exist_ok=True)
os.makedirs('saved_agents', exist_ok=True)

# Default parameters
EPISODES = 100
MAX_STEPS = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 100000  
LEARNING_RATE = 0.00005 
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05 
EPSILON_DECAY = 0.998  
TARGET_UPDATE_FREQ = 100  
TRAIN_FREQ = 4
SAVE_FREQ = 10
ENTROPY_BETA = 0.01 

episode_rewards = []
episode_lengths = []
episode_max_x_positions = []
loss_values = []
epsilons = []
running_avg_rewards = []
episode_durations = []
milestone_counts = []
action_distributions = []

def plot_training_progress(save_path='graphs/dqn_progress.png', agent=None):
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards, label='Rewards')
    if len(running_avg_rewards) > 0:
        plt.plot(running_avg_rewards, label='Running Avg (20 episodes)', color='red', linewidth=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot max x positions
    plt.subplot(3, 2, 3)
    plt.plot(episode_max_x_positions)
    # Plot horizontal lines for significant positions
    if len(episode_max_x_positions) > 0:
        max_pos = max(episode_max_x_positions)
        for i in range(100, int(max_pos) + 100, 100):
            plt.axhline(y=i, color='r', linestyle='--', alpha=0.3)
    plt.title('Max X Position Reached')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.grid(True)
    
    # Plot loss values
    if len(loss_values) > 0:
        plt.subplot(3, 2, 4)
        plt.plot(loss_values)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale for better visibility
        plt.grid(True)
    
    # Plot epsilon decay
    plt.subplot(3, 2, 5)
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Plot action distribution (if agent is provided)
    if agent is not None and agent.total_actions > 0:
        plt.subplot(3, 2, 6)
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
        action_probs = agent.action_counts / agent.total_actions
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'mediumaquamarine', 'plum']
        plt.bar(action_names, action_probs, color=colors)
        plt.title('Action Distribution')
        plt.ylabel('Frequency')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
    elif len(action_distributions) > 0:
        plt.subplot(3, 2, 6)
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
        latest_dist = action_distributions[-1]
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'mediumaquamarine', 'plum']
        plt.bar(action_names, latest_dist, color=colors)
        plt.title('Action Distribution (Latest Episode)')
        plt.ylabel('Frequency')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Standard DQN Mario Training')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epsilon_decay', type=float, default=EPSILON_DECAY, help='Epsilon decay rate')
    parser.add_argument('--entropy_beta', type=float, default=ENTROPY_BETA, help='Entropy regularization weight')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--save_freq', type=int, default=SAVE_FREQ, help='How often to save model')
    parser.add_argument('--frame_skip', type=int, default=3, help='Number of frames to skip (1-4)')
    
    args = parser.parse_args()
    
    print("Starting Standard DQN training with action diversity and entropy regularization...")
    print(f"Learning rate: {args.lr}")
    print(f"Epsilon decay: {args.epsilon_decay}")
    print(f"Entropy beta: {args.entropy_beta}")
    print(f"Frame skip: {args.frame_skip}")
    
    agent = DQNAgent(input_size=1024, hidden_size=256, output_size=7, 
                    learning_rate=args.lr, gamma=GAMMA, 
                    entropy_beta=args.entropy_beta)
    
    memory = PrioritizedReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    
    epsilon = EPSILON_START
    total_steps = 0
    
    for episode in range(1, args.episodes + 1):
        try:
            env = create_mario_env(stack_frames=4)
            state = env.reset()
            
            episode_reward = 0
            episode_loss = []
            max_x_position = 0
            step = 0
            done = False
            episode_start_time = time.time()
            
            episode_action_counts = np.zeros(7)
            
            while not done and step < MAX_STEPS:
                action = agent.get_action(state, epsilon, enforce_diversity=True)
                episode_action_counts[action] += 1
                
                next_state, reward, done, info = env.step(action)
                
                if isinstance(info, dict) and 'x_pos' in info:
                    max_x_position = max(max_x_position, info['x_pos'])
                
                memory.add(state, action, reward, next_state, float(done))
                
                state = next_state
                episode_reward += reward
                step += 1
                total_steps += 1
                
                if total_steps % TRAIN_FREQ == 0 and len(memory) > BATCH_SIZE:
                    experiences = memory.sample()
                    loss, td_errors = agent.train(experiences)
                    episode_loss.append(loss)
                    
                    memory.update_priorities(experiences[6], td_errors.reshape(-1))
                
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    agent.hard_update_target_network()
                
                if args.render and episode % 10 == 0 and step % 4 == 0:
                    env.render()
            
            episode_duration = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            episode_max_x_positions.append(max_x_position)
            epsilons.append(epsilon)
            episode_durations.append(episode_duration)
            
            if step > 0:
                action_dist = episode_action_counts / step
                action_distributions.append(action_dist)
            
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                loss_values.append(avg_loss)
            else:
                avg_loss = None
            
            window_size = min(20, episode)
            if episode >= window_size:
                running_avg = sum(episode_rewards[-window_size:]) / window_size
                running_avg_rewards.append(running_avg)
            else:
                running_avg = sum(episode_rewards) / episode
                running_avg_rewards.append(running_avg)
            
            epsilon = max(EPSILON_END, epsilon * args.epsilon_decay)
            
            action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
            action_summary = " | ".join([f"{name}: {episode_action_counts[i]:.0f} ({(episode_action_counts[i]/step)*100:.1f}%)" 
                                        for i, name in enumerate(action_names) if episode_action_counts[i] > 0])
            
            print(f"Episode {episode}/{args.episodes} - "
                  f"Steps: {step}, Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.4f}, Max X: {max_x_position}, "
                  f"Loss: {avg_loss if avg_loss is not None else 'N/A'}")
            print(f"Actions: {action_summary}")
            
            if episode % args.save_freq == 0 or episode == args.episodes:
                agent.save(f'saved_agents/standard_dqn_ep_{episode}.pth')
                plot_training_progress(f'graphs/dqn_progress_ep_{episode}.png', agent)
            
            env.close()
            del env
            gc.collect()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            try:
                env.close()
            except:
                pass
            try:
                del env
            except:
                pass
            gc.collect()
            time.sleep(1.0)
    
    print("Standard DQN training complete!")
    agent.save('saved_agents/standard_dqn_final.pth')
    plot_training_progress('graphs/standard_dqn_final.png', agent)

if __name__ == "__main__":
    main()