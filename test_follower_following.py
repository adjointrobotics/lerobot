#!/usr/bin/env python

"""
Test script to verify the follower actually follows the leader's movements.
This tests the core teleoperation functionality.
"""

import time
from lerobot.common.robots import make_robot_from_config
from lerobot.common.teleoperators import make_teleoperator_from_config
from lerobot.common.utils.utils import init_logging

from lerobot.common.robots.widow_ai_follower.config_widow_ai_follower import WidowAIFollowerConfig
from lerobot.common.teleoperators.widow_ai_leader.config_widow_ai_leader import WidowAILeaderConfig


def test_following_behavior():
    """Test that the follower actually follows the leader."""
    
    # Configuration
    teleop_config = WidowAILeaderConfig(
        port="192.168.1.2",
        model="V0_LEADER"
    )
    
    robot_config = WidowAIFollowerConfig(
        port="192.168.1.3", 
        model="V0_FOLLOWER",
        cameras={}  # No cameras
    )
    
    init_logging()
    print("=" * 60)
    print("TESTING FOLLOWER FOLLOWING BEHAVIOR")
    print("=" * 60)
    
    teleop = make_teleoperator_from_config(teleop_config)
    robot = make_robot_from_config(robot_config)

    try:
        print("\n1. Initializing both arms...")
        teleop.connect()
        robot.connect()
        print("✅ Both arms initialized!")
        
        print("\n2. Setting teleoperation modes...")
        teleop.prepare_for_teleoperation()
        robot.prepare_for_teleoperation()
        print("✅ Leader is now movable by hand, follower should follow!")
        
        print("\n3. Starting teleoperation test...")
        print("MOVE THE LEADER ARM BY HAND - the follower should follow!")
        print("Test will run for 15 seconds...")
        
        start_time = time.time()
        loop_count = 0
        
        while time.time() - start_time < 15.0:
            loop_count += 1
            loop_start_time = time.perf_counter()
            
            # Get leader position
            leader_action = teleop.get_action()
             
            # Get follower forces for feedback (like working demo)
            try:
                follower_efforts = robot.get_external_efforts()
                # Send scaled force feedback to leader (small gain for stability)
                scaled_feedback = {}
                for motor_force_key, effort in follower_efforts.items():
                    scaled_feedback[motor_force_key] = -0.01 * effort  # Very small gain
                teleop.send_feedback(scaled_feedback)
            except:
                # If force feedback fails, just send zeros to keep leader stable
                pass
            
            # Send leader position to follower (simple, like before)
            sent_action = robot.send_action(leader_action)
            
            # Log every 50 loops (less frequent to reduce spam)
            if loop_count % 50 == 0:
                # Get follower current position for comparison
                follower_obs = robot.get_observation()
                follower_positions = {k: v for k, v in follower_obs.items() if k.endswith('.pos')}
                
                loop_time = (time.perf_counter() - loop_start_time) * 1000
                print(f"\n--- Loop {loop_count} (t={time.time() - start_time:.1f}s, {loop_time:.1f}ms) ---")
                print("LEADER positions:")
                for k, v in leader_action.items():
                    print(f"  {k}: {v:7.2f}")
                print("FOLLOWER positions:")
                for k, v in follower_positions.items():
                    print(f"  {k}: {v:7.2f}")
                
                # Check if follower is actually moving
                leader_vals = list(leader_action.values())
                follower_vals = list(follower_positions.values())
                if len(leader_vals) == len(follower_vals):
                    max_diff = max(abs(l - f) for l, f in zip(leader_vals, follower_vals))
                    print(f"Max position difference: {max_diff:.3f}")
                    if max_diff < 2.0:  # Within 2 degrees
                        print("✅ Follower tracking well!")
                    else:
                        print("⚠️  Large tracking error")
            
            # NO artificial timing - run at natural speed like working demo
        
        print(f"\n✅ Test completed! Ran {loop_count} loops.")
        print("Did the follower arm move when you moved the leader?")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDisconnecting...")
        try:
            teleop.disconnect()
        except:
            pass
        try:
            robot.disconnect()
        except:
            pass
        print("Done!")


if __name__ == "__main__":
    test_following_behavior() 