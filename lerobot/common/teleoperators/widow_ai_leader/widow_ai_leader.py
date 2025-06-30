#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors.trossen import TrossenArmDriver

from ..teleoperator import Teleoperator
from .config_widow_ai_leader import WidowAILeaderConfig

logger = logging.getLogger(__name__)


class WidowAILeader(Teleoperator):
    """Trossen Widow AI Leader Arm for teleoperation."""

    config_class = WidowAILeaderConfig
    name = "widow_ai_leader"

    def __init__(self, config: WidowAILeaderConfig):
        super().__init__(config)
        self.config = config
        
        # Use simplified TrossenArmDriver
        self.bus = TrossenArmDriver(
            port=self.config.port,
            model=self.config.model,
        )
        
        # Define motor names for compatibility
        self.motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_1", "wrist_2", "wrist_3", "gripper"]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.force": float for motor in self.motor_names}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(f"{self} is pre-calibrated, no calibration needed.")

    def configure(self) -> None:
        self.bus.initialize_for_teleoperation(is_leader=True)
        self.bus.set_teleoperation_mode(is_leader=True)

    def prepare_for_teleoperation(self) -> None:
        """Set the teleoperator to teleoperation mode after both arms are initialized."""
        self.bus.set_teleoperation_mode(is_leader=True)

    def setup_motors(self) -> None:
        logger.info(f"{self} motors are pre-configured.")

    def get_action(self) -> dict[str, float]:
        """Get the current joint positions from the leader arm."""
        start = time.perf_counter()
        positions = self.bus.read("Present_Position")
        
        # Convert numpy array to dict with motor names
        action = {}
        for i, motor in enumerate(self.motor_names):
            if i < len(positions):
                action[f"{motor}.pos"] = float(positions[i])
            else:
                action[f"{motor}.pos"] = 0.0
                
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send force feedback to the leader arm."""
        force_feedback = []
        for motor in self.motor_names:
            force_key = f"{motor}.force"
            if force_key in feedback:
                force_feedback.append(-1 * self.config.force_feedback_gain * feedback[force_key])
            else:
                force_feedback.append(0.0)
        
        if force_feedback:
            self.bus.write("External_Efforts", force_feedback)
            logger.debug(f"{self} sent force feedback: {force_feedback}")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
