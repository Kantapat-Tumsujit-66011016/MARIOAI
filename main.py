import numpy as np
import time
import cv2
import argparse
import gc
from improved_env import create_improved_mario_env
from standard_dqn import DQNAgent

def test_agent(model_path='saved_agents/standard_dqn_final.pth', 
                                max_steps=10000, 
                                render_delay=5,
                                start_position=None):
    try:
        env = create_improved_mario_env(stack_frames=4, restrict_left=False)
        state = env.reset()
        
        agent = DQNAgent(input_size=1024, hidden_size=256, output_size=7)
        success = agent.load(model_path)
        
        if not success:
            print(f"Failed to load model from {model_path}")
            return
            
        print(f"Loaded model from {model_path}")
        print("Controls: Press 'q' to quit, 's' to slow down, 'f' to speed up")
        
        current_x = 0
        info = {'x_pos': 0, 'y_pos': 0}  
        
        if start_position is not None:
            
            while current_x < start_position:
                if np.random.random() < 0.7:
                    action = 3  
                else:
                    action = 4  
                
                next_state, _, done, info = env.step(action)
                
                if done:
                    state = env.reset()
                    continue
                
                state = next_state
                if isinstance(info, dict) and 'x_pos' in info:
                    current_x = info['x_pos']
                
                env.render()
                time.sleep(0.001)  
            
        
        # Run the agent with obstacle assistance
        done = False
        max_x_position = current_x
        last_y = 0
        stuck_counter = 0
        last_positions = []
        step = 0
        
        obstacles_cleared = 0
        
        action_counts = np.zeros(7)
        
        in_intervention = False
        current_sequence = []
        sequence_index = 0
        intervention_cooldown = 0
        
        jump_patterns = [
            [3, 3, 4, 4, 4],
            
            [3, 3, 3, 4, 4, 4, 4],
            
            [0, 4, 4, 4, 4, 4, 4],
            
            [3, 3, 3, 3, 4, 4, 4, 4, 4],
            
            [3, 3, 4, 4, 4, 3, 3, 4, 4, 4],
            
            [6, 6, 0, 3, 3, 3, 4, 4, 4, 4, 4]
        ]
        
        extreme_patterns = [
            [6, 6, 6, 6, 6, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
            
            [6, 6, 6, 6, 6, 6, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
            
            [3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4],
            
            [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            
            [6, 6, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        ]
        
        precision_patterns = [
            [3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
            [3, 3, 3, 3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 2],
            [6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        ]
        
        position_722_patterns = [
            [6, 6, 6, 0, 0, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            
            [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            
            [3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            
            [6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
]
        
        pattern_index = 0
        total_attempts = 0
        last_obstacle_x = 0
        using_extreme_patterns = False
        using_precision_patterns = False
        using_position_722_patterns = False
        
        while not done and step < max_steps:
            if isinstance(info, dict):
                if 'x_pos' in info:
                    current_x = info['x_pos']
                if 'y_pos' in info:
                    last_y = info['y_pos']
            
            last_positions.append(current_x)
            if len(last_positions) > 30:
                last_positions.pop(0)
            
            if current_x > max_x_position:
                progress = current_x - max_x_position
                max_x_position = current_x
                
                if progress > 20 and stuck_counter > 20:
                    obstacles_cleared += 1
                    stuck_counter = 0
                    
                    pattern_index = 0
                    total_attempts = 0
                    using_extreme_patterns = False
                    using_precision_patterns = False
                    using_position_722_patterns = False
            
            stuck = False
            if len(last_positions) >= 20:
                position_range = max(last_positions) - min(last_positions)
                if position_range < 5:
                    stuck = True
                    stuck_counter += 1
                else:
                    stuck_counter = max(0, stuck_counter - 1)
            
            if 715 <= current_x <= 725 and stuck:
                using_position_722_patterns = True

            if 897 <= current_x <= 899 and stuck:
                using_position_722_patterns = True
            
            if intervention_cooldown > 0:
                intervention_cooldown -= 1
            
            if in_intervention:
                if sequence_index < len(current_sequence):
                    action = current_sequence[sequence_index]
                    sequence_index += 1
                else:
                    in_intervention = False
                    intervention_cooldown = 20
                    
                    if current_x > last_obstacle_x + 20:
                        pattern_index = 0 
                        total_attempts = 0
                        using_extreme_patterns = False
                        using_precision_patterns = False
                    else:
                        total_attempts += 1
                        pattern_index = (pattern_index + 1) 
                        
                        if total_attempts > len(jump_patterns) and not using_extreme_patterns:
                            using_extreme_patterns = True
                            pattern_index = 0
                        
                        if using_extreme_patterns and pattern_index >= len(extreme_patterns):
                            if not using_precision_patterns:
                                using_precision_patterns = True
                                pattern_index = 0
                            else:
                                pattern_index = pattern_index % len(precision_patterns)
            
            elif stuck and stuck_counter > 20 and not intervention_cooldown:
                
                if using_position_722_patterns:
                    pattern_set = position_722_patterns
                    pattern_name = "position 722"
                elif using_precision_patterns:
                    pattern_set = precision_patterns
                    pattern_name = "precision"
                elif using_extreme_patterns:
                    pattern_set = extreme_patterns
                    pattern_name = "extreme"
                else:
                    pattern_set = jump_patterns
                    pattern_name = "standard"
                
                pattern_index = pattern_index % len(pattern_set)
                current_sequence = pattern_set[pattern_index].copy()
                
                sequence_index = 0
                in_intervention = True
                
                last_obstacle_x = current_x
                
                action = current_sequence[sequence_index]
                sequence_index += 1
                
                if stuck_counter > 200 and total_attempts > 15:
                    current_sequence = [6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                    action = current_sequence[0]
                    sequence_index = 1
            
            else:
                epsilon = 0.05 if not stuck else 0.1  
                action = agent.get_action(state, epsilon=epsilon)
            
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            step += 1
            
            env.render()
            time.sleep(render_delay)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('s'):
                render_delay = min(0.1, render_delay + 0.01)
                print(f"Speed: {1/render_delay:.1f}x")
            elif key == ord('f'): 
                render_delay = max(0.001, render_delay - 0.005)
                print(f"Speed: {1/render_delay:.1f}x")
            elif key == ord('r'):
                print("Manual reset requested")
                if last_obstacle_x > 0:
                    reset_pos = max(0, last_obstacle_x - 30)
                    
                    env.close()
                    env = create_improved_mario_env(stack_frames=4, restrict_left=False)
                    state = env.reset()
                    
                    current_x = 0
                    while current_x < reset_pos:
                        action = 3 
                        state, _, done, info = env.step(action)
                        if done:
                            state = env.reset()
                            continue
                        
                        if isinstance(info, dict) and 'x_pos' in info:
                            current_x = info['x_pos']
                        
                        env.render()
                        time.sleep(0.001)
                    
                    
                    in_intervention = False
                    intervention_cooldown = 0
                    stuck_counter = 0
            
        # Final report
        print("\n=== Run Summary ===")
        print(f"Steps taken: {step}")
        print(f"Max position reached: {max_x_position}")
        print(f"Obstacles cleared: {obstacles_cleared}")
        
        print("\nAction distribution:")
        action_names = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
        for i, count in enumerate(action_counts):
            percentage = (count / step) * 100 if step > 0 else 0
            print(f"  {action_names[i]}: {count:.0f} ({percentage:.1f}%)")
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mario agent with enhanced obstacle assistance')
    parser.add_argument('--model', type=str, default='saved_agents/standard_dqn_final.pth',
                      help='Path to saved model file')
    parser.add_argument('--steps', type=int, default=10000,
                      help='Maximum steps to run')
    parser.add_argument('--delay', type=float, default=0.01,
                      help='Rendering delay')
    parser.add_argument('--start', type=int, default=None,
                      help='Starting position (to skip ahead)')
    parser.add_argument('--pos722', action='store_true',
                      help='Focus on position 722 obstacle')
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        max_steps=args.steps,
        render_delay=args.delay,
        start_position=args.start
    )