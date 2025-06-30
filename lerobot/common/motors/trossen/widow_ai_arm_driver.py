import time
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np
import trossen_arm as trossen


class TrossenArmDriver:
    """
    Simple, direct interface for Trossen Robotics' robotic arms.
    
    This is a simplified implementation that directly uses the trossen_arm library.
    """

    def __init__(
        self,
        port: str,
        model: str = "V0_LEADER",
        mock: bool = False,
    ):
        self.port = port
        self.model = model
        self.mock = mock
        self.driver = None
        self.is_connected = False
        self.logs = {}
        self.fps = 30
        
        # Home and sleep poses (unused for now)
        self.home_pose = [0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0]
        self.sleep_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.MIN_TIME_TO_MOVE = 3.0 / self.fps

    @property
    def is_connected(self) -> bool:
        """Check if the Trossen arm is connected."""
        return self._is_connected and self.driver is not None

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value

    def _get_model_config(self):
        """Get the model configuration using the actual enum objects."""
        trossen_arm_models = {
            "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
            "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
        }
        
        try:
            return trossen_arm_models[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e

    def connect(self) -> None:
        """Connect to the Trossen arm."""
        if self.is_connected:
            print(f"TrossenArmDriver({self.port}) is already connected.")
            return

        print(f"Connecting to {self.model} arm at {self.port}...")

        # Initialize the driver
        self.driver = trossen.TrossenArmDriver()

        # Get the model configuration
        try:
            model_name, model_end_effector = self._get_model_config()
        except Exception as e:
            raise e

        print("Configuring the drivers...")

        # Configure the driver
        try:
            self.driver.configure(model_name, model_end_effector, self.port, False)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to configure the driver for the {self.model} arm at {self.port}.")
            raise e

        # Mark as connected
        self.is_connected = True

    def disconnect(self) -> None:
        """Disconnect from the Trossen arm."""
        if not self.is_connected:
            return
        self.is_connected = False
        print(f"{self.model} arm disconnected.")

    def read(self, data_name: str) -> np.ndarray:
        """Read data from the Trossen arm."""
        if not self.is_connected:
            raise RuntimeError(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")

        start_time = time.perf_counter()

        if data_name == "Present_Position":
            values = self.driver.get_all_positions()
        elif data_name == "External_Efforts":
            values = self.driver.get_all_external_efforts()
        else:
            raise ValueError(f"Data name: {data_name} is not supported for reading.")

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time
        return np.array(values, dtype=np.float32)

    def write(self, data_name: str, values: Union[float, List[float], np.ndarray]) -> None:
        """Write data to the Trossen arm."""
        if not self.is_connected:
            raise RuntimeError(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")

        start_time = time.perf_counter()

        if data_name == "Goal_Position":
            values = np.array(values, dtype=np.float32)
            self.driver.set_all_positions(values.tolist(), self.MIN_TIME_TO_MOVE, False)
        elif data_name == "Torque_Enable":
            if values == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                self.driver.set_all_modes(trossen.Mode.external_effort)
        elif data_name == "External_Efforts":
            values = np.array(values, dtype=np.float32)
            self.driver.set_all_external_efforts(values.tolist(), 0.0, False)
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.velocity)
            self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, False)
        else:
            raise ValueError(f"Data name: {data_name} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def initialize_for_teleoperation(self, is_leader: bool = True) -> None:
        """Initialize the arm for teleoperation."""
        print(f"Initializing {self.model} arm for teleoperation...")
        
        # Don't move to home position - just set the mode like SO100/SO101
        print(f"Setting {self.model} arm mode for teleoperation...")
        if is_leader:
            self.driver.set_all_modes(trossen.Mode.external_effort)
            # Send zero forces immediately to prevent instability
            print(f"Sending zero forces to {self.model} arm for stability...")
            try:
                zero_forces = [0.0] * self.driver.get_num_joints()
                self.driver.set_all_external_efforts(zero_forces, 0.0, False)
                print(f"Zero forces sent to {self.model} arm - should be stable now")
            except Exception as e:
                print(f"WARNING: Failed to send zero forces to {self.model} arm: {e}")
        else:
            self.driver.set_all_modes(trossen.Mode.position)
            
        print(f"{self.model} arm initialization complete!")

    def set_teleoperation_mode(self, is_leader: bool = True) -> None:
        """Set the appropriate mode for teleoperation."""
        if is_leader:
            print(f"Setting {self.model} arm to external effort mode for teleoperation...")
            self.driver.set_all_modes(trossen.Mode.external_effort)
            
            # Send zero forces immediately to prevent instability
            print(f"Sending zero forces to {self.model} arm for stability...")
            try:
                zero_forces = [0.0] * self.driver.get_num_joints()
                self.driver.set_all_external_efforts(zero_forces, 0.0, False)
                print(f"Zero forces sent to {self.model} arm - should be stable now")
            except Exception as e:
                print(f"WARNING: Failed to send zero forces to {self.model} arm: {e}")
        else:
            print(f"Setting {self.model} arm to position mode for teleoperation...")
            self.driver.set_all_modes(trossen.Mode.position)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            if getattr(self, "_is_connected", False):
                self.disconnect()
        except Exception as e:
            print(f"Warning: Exception during cleanup: {e}")