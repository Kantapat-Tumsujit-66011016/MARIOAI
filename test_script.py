import numpy as np
import time
import cv2
import argparse
import gc
from improved_env import create_improved_mario_env
from standard_dqn import DQNAgent

def test_agent(model_path='saved_agents/standard_dqn_final.pth', max_steps=10000, 
               add_exploration=False, explore_rate=0.05, allow_left=False):

    try:
        env = create_improved_mario_env(stack_frames=4, 
                                       restrict_left=not allow_left,
                                       stuck_penalty=False) 
        state = env.reset()
        
        agent = DQNAgent(input_size=1024, hidden_size=256, 
                        output_size=6 if not allow_left else 7)
        loaded = agent.load(model_path)
        
        if not loaded:
            print(f"Could not load model from {model_path}. Exiting.")
            return
        
        print(f"Testing agent from {model_path}")
        print("Controls: Press 'q' to quit, 's' to slow down, 'f' to speed up")
        if add_exploration:
            print(f"Using exploration during testing with epsilon={explore_rate}")
        
        state = env.reset()
        total_reward = 0
        max_x_position = 0
        step = 0
        done = False
        
        positions = []
        rewards = []
        actions_taken = np.zeros(6 if not allow_left else 7)
        milestone_reached = set()
        delay = 0.01  
        
        start_time = time.time()
        
        while not done and step < max_steps:
            if add_exploration:
                action = agent.get_action(state, epsilon=explore_rate, enforce_diversity=True)
            else:
                action = agent.get_action(state, epsilon=0.0)
                
            actions_taken[action] += 1
            
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            if isinstance(info, dict) and 'x_pos' in info:
                current_pos = info['x_pos']
                positions.append(current_pos)
                max_x_position = max(max_x_position, current_pos)
                
                milestone = (current_pos // 100) * 100
                if milestone > 0:
                    milestone_reached.add(milestone)
            
            rewards.append(reward)
            
            state = next_state
            
            env.render()
            time.sleep(delay)
            
        duration = time.time() - start_time
        
        print("\n=== Agent Performance Summary ===")
        print(f"Steps taken: {step}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Maximum position reached: {max_x_position}")
        print(f"Episode duration: {duration:.2f} seconds")
        print(f"Milestones reached: {sorted(milestone_reached)}")
        
        print("\nAction distribution:")
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A"]
        if allow_left:
            action_names.append("LEFT")
            
        for i, count in enumerate(actions_taken):
            percentage = (count / step) * 100 if step > 0 else 0
            print(f"  {action_names[i]}: {count:.0f} ({percentage:.1f}%)")
        
        if rewards:
            print("\nReward statistics:")
            print(f"  Average reward per step: {sum(rewards)/len(rewards):.4f}")
            print(f"  Max reward in a step: {max(rewards):.2f}")
            print(f"  Min reward in a step: {min(rewards):.2f}")
            
            if len(positions) > 1:
                progress_rate = (positions[-1] - positions[0]) / len(positions)
                print(f"  Forward progress rate: {progress_rate:.4f} units/step")
        
        env.close()
        del env
        gc.collect()
        
        return {
            "steps": step,
            "reward": total_reward,
            "max_position": max_x_position,
            "duration": duration,
            "milestones": milestone_reached,
            "actions": actions_taken
        }
    
    except Exception as e:
        print(f"Error during testing: {e}")
        try:
            env.close()
        except:
            pass
        try:
            del env
        except:
            pass
        gc.collect()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained DQN Mario agent')
    parser.add_argument('--model', type=str, default='saved_agents/standard_dqn_final.pth', 
                      help='Path to the saved model file')
    parser.add_argument('--steps', type=int, default=10000, 
                      help='Maximum number of steps to run')
    parser.add_argument('--explore', action='store_true',
                      help='Add some exploration during testing')
    parser.add_argument('--epsilon', type=float, default=0.05,
                      help='Exploration rate if --explore is enabled')
    parser.add_argument('--allow_left', action='store_true',
                      help='Allow LEFT actions during testing')
    
    args = parser.parse_args()
    test_agent(model_path=args.model, max_steps=args.steps, 
               add_exploration=args.explore, explore_rate=args.epsilon,
               allow_left=args.allow_left)