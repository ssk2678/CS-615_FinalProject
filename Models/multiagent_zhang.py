import numpy as np


DETECTION_RATE = 0.6
MAX_SPEED = 13.89


def zhang_reward(traffic_signal):
    """Return the Zhang-style speed reward for one traffic signal."""
    vehicle_speeds = []
    for lane in traffic_signal.lanes:
        for vehicle_id in traffic_signal.sumo.lane.getLastStepVehicleIDs(lane):
            if np.random.random() <= DETECTION_RATE:
                vehicle_speeds.append(traffic_signal.sumo.vehicle.getSpeed(vehicle_id))

    detected_speed = np.mean(vehicle_speeds) if vehicle_speeds else MAX_SPEED
    detected_speed = min(float(detected_speed), MAX_SPEED)
    return (-1.0 / MAX_SPEED) * (MAX_SPEED - detected_speed)


def mask_observation(observation, action_count):
    """Hide undetected lanes in one observation vector."""
    masked = np.array(observation, dtype=np.float32, copy=True)
    prefix_size = action_count + 1
    lane_count = (masked.size - prefix_size) // 2
    visible_lanes = np.random.random(lane_count) <= DETECTION_RATE

    masked[prefix_size : prefix_size + lane_count] *= visible_lanes
    masked[prefix_size + lane_count : prefix_size + 2 * lane_count] *= visible_lanes
    return masked


def mask_observations(observation_dict, env):
    """Hide undetected lanes while preserving phase and minimum-green features."""
    masked_observations = {}
    for traffic_light_id, observation in observation_dict.items():
        action_count = env.action_space(traffic_light_id).n
        masked_observations[traffic_light_id] = mask_observation(observation, action_count)

    return masked_observations
