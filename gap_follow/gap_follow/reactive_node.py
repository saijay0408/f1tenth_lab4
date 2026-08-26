#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.subscription = self.create_subscription(
            LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        self.max_range = 3.0
        self.smooth_window = 5

        self.bubble_radius = 0.9
        self.min_gap_depth = 0.6

        self.angle_min = None
        self.angle_increment = None

        self.max_steer = np.radians(24.0)

    def preprocess_lidar(self, ranges):
        proc_ranges = np.array(ranges, dtype=np.float64)
        proc_ranges = np.nan_to_num(proc_ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_range)

        kernel = np.ones(self.smooth_window) / self.smooth_window
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        is_free = free_space_ranges > self.min_gap_depth

        best_start, best_len = -1, 0
        cur_start = -1
        for i, free in enumerate(is_free):
            if free:
                if cur_start == -1:
                    cur_start = i
            else:
                if cur_start != -1:
                    length = i - cur_start
                    if length > best_len:
                        best_len = length
                        best_start = cur_start
                    cur_start = -1
        if cur_start != -1:
            length = len(is_free) - cur_start
            if length > best_len:
                best_len = length
                best_start = cur_start

        if best_start == -1:
            return 0, len(free_space_ranges) - 1

        return best_start, best_start + best_len - 1

    def find_best_point(self, start_i, end_i, ranges):
        segment = ranges[start_i:end_i + 1]
        max_val = np.max(segment)
        deep_indices = np.where(segment >= max_val - 1e-3)[0]
        furthest_center = start_i + int(np.mean(deep_indices))
        gap_midpoint = (start_i + end_i) // 2

        return (furthest_center + gap_midpoint) // 2

    def lidar_callback(self, data):
        self.angle_min = data.angle_min
        self.angle_increment = data.angle_increment

        proc_ranges = self.preprocess_lidar(data.ranges)

        closest_idx = int(np.argmin(proc_ranges))
        closest_dist = max(proc_ranges[closest_idx], 1e-3)

        angle_radius = np.arctan2(self.bubble_radius, closest_dist)
        idx_radius = max(int(angle_radius / self.angle_increment), 1)
        lo = max(0, closest_idx - idx_radius)
        hi = min(len(proc_ranges), closest_idx + idx_radius + 1)
        proc_ranges[lo:hi] = 0.0

        start_i, end_i = self.find_max_gap(proc_ranges)
        best_idx = self.find_best_point(start_i, end_i, proc_ranges)

        steering_angle = self.angle_min + best_idx * self.angle_increment
        steering_angle = float(np.clip(steering_angle, -self.max_steer, self.max_steer))

        abs_deg = abs(np.degrees(steering_angle))
        if abs_deg <= 10.0:
            speed = 1.2
        elif abs_deg <= 20.0:
            speed = 0.8
        else:
            speed = 0.4

        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher_.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
