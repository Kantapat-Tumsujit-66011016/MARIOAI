import numpy as np
import time
import os
import gc
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime
import argparse
from improved_env import create_improved_mario_env
from standard_dqn import DQNAgent
from prioritized_replay import PrioritizedReplayBuffer

os.makedirs('graphs', exist_ok=True)
os.makedirs('saved_agents', exist_ok=True)
os.makedirs('training_logs', exist_ok=True)

EPISODES = 500
MAX_STEPS = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 500000
LEARNING_RATE = 0.00005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQ = 200
TRAIN_FREQ = 4
SAVE_FREQ = 25
ENTROPY_BETA = 0.015 
LR_DECAY_FREQ = 50
LR_DECAY_RATE = 0.9

metrics = {
    'episode_rewards': [],
    'episode_lengths': [],
    'episode_max_x_positions': [],
    'loss_values': [],
    'epsilons': [],
    'running_avg_rewards': [],
    'episode_durations': [],
    'learning_rates': [],
    'action_distributions': []
}

def plot_training_progress(save_path, agent=None, episode=0):
    """Plot comprehensive training metrics"""
    plt.figure(figsize=(20, 15))
    
    # Plot rewards with milestones
    plt.subplot(3, 3, 1)
    plt.plot(metrics['episode_rewards'], label='Rewards')
    if len(metrics['running_avg_rewards']) > 0:
        plt.plot(metrics['running_avg_rewards'], label='Running Avg (20 ep)', color='red', linewidth=2)
    
    # Add episode markers for milestones
    max_positions = metrics['episode_max_x_positions']
    milestones = set()
    for i, pos in enumerate(max_positions):
        milestone = (pos // 100) * 100
        if milestone > 0 and milestone not in milestones:
            milestones.add(milestone)
            plt.axvline(x=i, color='g', linestyle='--', alpha=0.5)
            plt.text(i, min(metrics['episode_rewards']), f"{milestone}", fontsize=8)
    
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(3, 3, 2)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot max x positions
    plt.subplot(3, 3, 3)
    plt.plot(metrics['episode_max_x_positions'])
    if len(metrics['episode_max_x_positions']) > 0:
        max_pos = max(metrics['episode_max_x_positions'])
        for i in range(100, int(max_pos) + 100, 100):
            plt.axhline(y=i, color='r', linestyle='--', alpha=0.3)
    plt.title('Max X Position Reached')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.grid(True)
    
    # Plot loss values
    if len(metrics['loss_values']) > 0:
        plt.subplot(3, 3, 4)
        plt.plot(metrics['loss_values'])
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.yscale('log') 
        plt.grid(True)
    
    # Plot epsilon decay
    plt.subplot(3, 3, 5)
    plt.plot(metrics['epsilons'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Plot learning rate
    if len(metrics['learning_rates']) > 0:
        plt.subplot(3, 3, 6)
        plt.plot(metrics['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.grid(True)
    
    # Plot action distribution (if agent is provided)
    if agent is not None and agent.total_actions > 0:
        plt.subplot(3, 3, 7)
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
        if len(action_names) > len(agent.action_counts):
            action_names = action_names[:len(agent.action_counts)]
        action_probs = agent.action_counts / agent.total_actions
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'mediumaquamarine', 'plum'][:len(action_probs)]
        plt.bar(action_names, action_probs, color=colors)
        plt.title('Overall Action Distribution')
        plt.ylabel('Frequency')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
    
    # Plot recent position progress
    if len(metrics['episode_max_x_positions']) > 0:
        plt.subplot(3, 3, 8)
        window = min(50, len(metrics['episode_max_x_positions']))
        recent_positions = metrics['episode_max_x_positions'][-window:]
        plt.plot(range(episode - window + 1, episode + 1), recent_positions)
        plt.title(f'Recent Position Progress (Last {window} Episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Max X Position')
        plt.grid(True)
    
    # Plot recent action distribution
    if len(metrics['action_distributions']) > 0:
        plt.subplot(3, 3, 9)
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A"]
        if len(metrics['action_distributions'][-1]) > len(action_names):
            action_names.append("LEFT")
        latest_dist = metrics['action_distributions'][-1]
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'mediumaquamarine', 'plum'][:len(latest_dist)]
        plt.bar(action_names, latest_dist, color=colors)
        plt.title('Recent Action Distribution')
        plt.ylabel('Frequency')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(filename='training_logs/metrics.json'):
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            if len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    serializable_metrics[key] = [arr.tolist() for arr in value]
                elif isinstance(value[0], (np.int32, np.int64, np.float32, np.float64)):
                    serializable_metrics[key] = [float(x) if isinstance(x, (np.float32, np.float64)) else int(x) for x in value]
                else:
                    serializable_metrics[key] = value
            else:
                serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f)

def load_metrics(filename='training_logs/metrics.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                loaded_metrics = json.load(f)
            
            for key, value in loaded_metrics.items():
                if key == 'action_distributions' and len(value) > 0:
                    metrics[key] = [np.array(arr) for arr in value]
                else:
                    metrics[key] = value
            
            return True
        except json.JSONDecodeError:
            print("Warning: Metrics file is corrupted. Starting with fresh metrics.")
            try:
                os.remove(filename)
            except:
                pass
            return False
    return False

def find_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir('saved_agents') if f.startswith('long_dqn_ep_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None, 0
    
    episode_numbers = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    if not episode_numbers:
        return None, 0
    
    max_episode = max(episode_numbers)
    latest_file = f'saved_agents/long_dqn_ep_{max_episode}.pth'
    
    return latest_file, max_episode

def main():
    parser = argparse.ArgumentParser(description='Long-term DQN Mario Training (500+ episodes)')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Initial learning rate')
    parser.add_argument('--epsilon_decay', type=float, default=EPSILON_DECAY, help='Epsilon decay rate')
    parser.add_argument('--entropy_beta', type=float, default=ENTROPY_BETA, help='Entropy regularization weight')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--save_freq', type=int, default=SAVE_FREQ, help='How often to save model')
    parser.add_argument('--allow_left', action='store_true', help='Allow LEFT actions')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting Long-Term DQN training for 500+ episodes")
    print("=" * 60)
    print(f"Initial learning rate: {args.lr}")
    print(f"Epsilon decay: {args.epsilon_decay}")
    print(f"Entropy beta: {args.entropy_beta}")
    print(f"LEFT actions: {'Allowed' if args.allow_left else 'Restricted'}")
    print(f"Checkpointing frequency: Every {args.save_freq} episodes")
    print("-" * 60)
    
    start_episode = 1
    if args.resume:
        latest_checkpoint, checkpoint_episode = find_latest_checkpoint()
        if latest_checkpoint and checkpoint_episode > 0:
            print(f"Resuming training from episode {checkpoint_episode}")
            print(f"Loading checkpoint: {latest_checkpoint}")
            start_episode = checkpoint_episode + 1
            
            metrics_loaded = load_metrics()
            if metrics_loaded:
                print("Loaded previous training metrics")
            else:
                print("Starting with fresh metrics tracking")
                if checkpoint_episode > 0:
                    metrics['episode_rewards'] = [0] * checkpoint_episode
                    metrics['episode_lengths'] = [0] * checkpoint_episode
                    metrics['episode_max_x_positions'] = [0] * checkpoint_episode
                    metrics['epsilons'] = [0] * checkpoint_episode
                    metrics['episode_durations'] = [0] * checkpoint_episode
                    metrics['learning_rates'] = [0] * checkpoint_episode
        else:
            print("No checkpoint found, starting fresh training")
    
    agent = DQNAgent(input_size=1024, hidden_size=256, output_size=6 if not args.allow_left else 7, 
                    learning_rate=args.lr, gamma=GAMMA, 
                    entropy_beta=args.entropy_beta)
    
    if args.resume and latest_checkpoint and checkpoint_episode > 0:
        agent.load(latest_checkpoint)
    
    memory = PrioritizedReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    
    epsilon = EPSILON_START
    if start_episode > 1:
        epsilon = max(EPSILON_END, EPSILON_START * (args.epsilon_decay ** (start_episode - 1)))
    
    learning_rate = args.lr
    total_steps = 0
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = f"training_logs/long_dqn_training_{current_time}.log"
    with open(log_file, 'w') as f:
        f.write(f"Long-term DQN Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Episodes: {args.episodes}, Start Episode: {start_episode}\n")
        f.write(f"Learning Rate: {args.lr}, Epsilon Decay: {args.epsilon_decay}\n")
        f.write(f"LEFT Actions: {'Allowed' if args.allow_left else 'Restricted'}\n")
        f.write("=" * 80 + "\n\n")
    
    try:
        for episode in range(start_episode, args.episodes + 1):
            episode_start_time = time.time()
            
            env = create_improved_mario_env(stack_frames=4, 
                                           restrict_left=not args.allow_left,
                                           stuck_penalty=True)
            state = env.reset()
            
           
            if np.random.random() < 0.2:  
                for _ in range(3):
                    if np.random.random() < 0.5:
                        boost_action = 3 
                    else:
                        boost_action = 4 
                    
                    next_state, _, _, _ = env.step(boost_action)
                    state = next_state
            
            episode_reward = 0
            episode_loss = []
            max_x_position = 0
            step = 0
            done = False
            
            episode_action_counts = np.zeros(6 if not args.allow_left else 7)
            
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
                
                if args.render and episode % 50 == 0 and step % 8 == 0:
                    env.render()
            
            episode_duration = time.time() - episode_start_time
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(step)
            metrics['episode_max_x_positions'].append(max_x_position)
            metrics['epsilons'].append(epsilon)
            metrics['episode_durations'].append(episode_duration)
            metrics['learning_rates'].append(learning_rate)
            
            if step > 0:
                action_dist = episode_action_counts / step
                metrics['action_distributions'].append(action_dist)
            
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                metrics['loss_values'].append(avg_loss)
            else:
                avg_loss = None
            
            window_size = min(20, len(metrics['episode_rewards']))
            running_avg = sum(metrics['episode_rewards'][-window_size:]) / window_size
            metrics['running_avg_rewards'].append(running_avg)
            
            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * args.epsilon_decay)
            
            # Decay learning rate
            if episode % LR_DECAY_FREQ == 0 and episode > 1:
                learning_rate *= LR_DECAY_RATE
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print(f"Learning rate decayed to: {learning_rate:.6f}")
            
            action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A"]
            if args.allow_left:
                action_names.append("LEFT")
                
            action_summary = " | ".join([f"{name}: {episode_action_counts[i]:.0f} ({(episode_action_counts[i]/step)*100:.1f}%)" 
                                        for i, name in enumerate(action_names) if episode_action_counts[i] > 0])
            
            print(f"Episode {episode}/{args.episodes} - "
                  f"Steps: {step}, Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.4f}, Max X: {max_x_position}, "
                  f"Loss: {avg_loss if avg_loss is not None else 'N/A'}")
            print(f"Actions: {action_summary}")
            
            with open(log_file, 'a') as f:
                f.write(f"Episode {episode} - Reward: {episode_reward:.2f}, Steps: {step}, Max X: {max_x_position}\n")
                f.write(f"Actions: {action_summary}\n")
                f.write(f"Duration: {episode_duration:.2f}s, Epsilon: {epsilon:.4f}, LR: {learning_rate:.6f}\n\n")
            
            if episode % args.save_freq == 0 or episode == args.episodes:
                checkpoint_path = f'saved_agents/long_dqn_ep_{episode}.pth'
                agent.save(checkpoint_path)
                plot_training_progress(f'graphs/long_dqn_ep_{episode}.png', agent, episode)
                save_metrics()
                print(f"Checkpoint saved at episode {episode}")
            
            env.close()
            del env
            gc.collect()
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save('saved_agents/long_dqn_interrupted.pth')
        plot_training_progress('graphs/long_dqn_interrupted.png', agent, episode)
        save_metrics()
        print("Checkpoint saved")
    except Exception as e:
        print(f"Error during training: {e}")
        try:
            agent.save('saved_agents/standard_dqn_emergency.pth')
            plot_training_progress('graphs/standard_dqn_emergency.png', agent, episode)
            save_metrics()
            print("Emergency checkpoint saved")
        except:
            print("Could not save emergency checkpoint")
    
    print("\nLong-term DQN training complete!")
    agent.save('saved_agents/standard_dqn_final.pth')
    plot_training_progress('graphs/standard_dqn_final.png', agent, args.episodes)
    save_metrics()
    
    max_position = max(metrics['episode_max_x_positions']) if metrics['episode_max_x_positions'] else 0
    final_avg_reward = metrics['running_avg_rewards'][-1] if metrics['running_avg_rewards'] else 0
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes completed: {len(metrics['episode_rewards'])}")
    print(f"Maximum position reached: {max_position}")
    print(f"Final average reward (20 ep): {final_avg_reward:.2f}")
    print(f"Final exploration rate: {epsilon:.4f}")
    print(f"Final learning rate: {learning_rate:.6f}")
    print("-" * 60)

if __name__ == "__main__":
    main()